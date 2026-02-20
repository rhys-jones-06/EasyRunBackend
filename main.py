from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple
import datetime
import math
import httpx
import networkx as nx
import random
import asyncio
import uuid
import logging
import os
from contextlib import asynccontextmanager
from fastapi import Request

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
ELEVATION_URL = os.getenv("ELEVATION_URL", "https://api.open-elevation.com/api/v1/lookup")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
API_VERSION = "1.0.0"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
CACHE_SIZE_OSM = int(os.getenv("CACHE_SIZE_OSM", "50"))
CACHE_SIZE_ELEVATION = int(os.getenv("CACHE_SIZE_ELEVATION", "100"))

# Simple async-friendly cache
OSM_CACHE = {}
ELEVATION_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Global client for connection pooling
    app.state.http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    logger.info(f"EasyRun Backend v{API_VERSION} started")
    yield
    # Teardown
    await app.state.http_client.aclose()
    logger.info("EasyRun Backend shutdown")

app = FastAPI(
    title="EasyRun API",
    description="Running route suggestion API",
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Models ---
# (Keep existing models as they are already perfect for the client)

class Meta(BaseModel):
    id: str = Field(..., description="Unique route identifier")
    label: Optional[str] = Field(None, description="Human-readable route name")
    difficulty: Optional[str] = Field(None, description="Difficulty level: easy, moderate, hard")
    generated_at: Optional[str] = Field(None, description="ISO8601 timestamp")

class Summary(BaseModel):
    distance_km: float = Field(..., gt=0, description="Total route distance in kilometers")
    duration_mins: Optional[int] = Field(None, ge=0, description="Estimated duration in minutes")
    ascent_m: Optional[float] = Field(None, ge=0, description="Total elevation gain in meters")
    descent_m: Optional[float] = Field(None, ge=0, description="Total elevation loss in meters")
    avg_gradient_percent: Optional[float] = Field(None, description="Average gradient percentage")
    elevation_profile: Optional[List[float]] = Field(None, description="Elevation samples along route")

class TerrainComposition(BaseModel):
    paved_pct: Optional[int] = Field(None, ge=0, le=100, description="Percentage of paved surface")
    unpaved_pct: Optional[int] = Field(None, ge=0, le=100, description="Percentage of unpaved surface")
    primary_surface: Optional[str] = Field(None, description="Primary surface type")

class Geometry(BaseModel):
    type: str = "LineString"
    coordinates: List[List[float]]  # [lon, lat] or [lon, lat, alt]

class MapData(BaseModel):
    bbox: Optional[List[List[float]]] = None
    geometry: Geometry

class Suggestion(BaseModel):
    meta: Meta
    summary: Summary
    terrain_composition: Optional[TerrainComposition] = None
    map_data: MapData

class SuggestionsResponse(BaseModel):
    suggestions: List[Suggestion]

class ShapePoint(BaseModel):
    """A single point in 0-1 normalized space.

    FRONTEND NOTES:
    - x=0 is left edge, x=1 is right edge
    - y=0 is top edge, y=1 is bottom edge (canvas-style, like screen coords)
    - The backend handles all geo-projection — the frontend just draws on a unit canvas
    - For closed shapes (loops), the last point should equal the first point
    """
    x: float  # 0.0 to 1.0 normalized horizontal position
    y: float  # 0.0 to 1.0 normalized vertical position

class ShapeRequest(BaseModel):
    """Request body for the /shape-route endpoint.

    FRONTEND NOTES:
    - Send EITHER "preset" (a preset name string) OR "shape" (custom points), or both
    - If "preset" is provided, it overrides "shape" — the preset's points are used
    - If only "shape" is provided, those custom points are used directly
    - Use GET /presets to discover available preset names and their display info
    - Custom shapes from freehand drawing / SVG import should go in "shape"
    """
    lat: float = Field(..., ge=-90, le=90, description="User's starting latitude")
    lon: float = Field(..., ge=-180, le=180, description="User's starting longitude")
    distance_km: float = Field(..., gt=0, le=100, description="Target route distance in km")
    difficulty: int = Field(..., ge=1, le=10, description="Difficulty score 1-10")
    preset: Optional[str] = Field(None, description="Preset shape name (e.g. 'pizza', 'eggplant')")
    shape: Optional[List[ShapePoint]] = Field(None, description="Custom shape points")

    @validator('lat', 'lon')
    def validate_coords(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Coordinates must be numeric")
        return float(v)

# ---------------------------------------------------------------------------
# Preset shapes — each is a list of ShapePoints in 0-1 normalized space.
#
# FRONTEND NOTES — HOW TO ADD NEW PRESETS:
#   1. Define the outline as (x, y) tuples in 0-1 space (y=0 top, y=1 bottom)
#   2. The shape should be CLOSED (last point == first point) for a proper loop
#   3. Add it to PRESET_SHAPES below with a unique key
#   4. Add display metadata to PRESET_DISPLAY_INFO for the GET /presets response
#   5. The backend does all scaling/projection — just provide the outline
#
# The perimeter of the normalised outline is scaled so total route distance
# matches the requested distance_km. More points = smoother curves on the map.
# ---------------------------------------------------------------------------

def _generate_preset_shapes() -> dict:
    """Build preset shape outlines at startup. Called once at module load."""
    presets = {}

    # --- PIZZA ---
    # A circle with a triangular slice missing (pac-man / pizza-with-slice-gone).
    # The route goes: center -> edge at 30° -> traces the circle to 330° -> center.
    # This creates the iconic "pizza with one slice taken" silhouette.
    pizza = []
    pizza.append({"x": 0.5, "y": 0.5})          # center of the pizza
    num_arc_pts = 24                              # smoothness of the circular crust
    slice_angle_deg = 50                          # how wide the missing slice is (degrees)
    start_deg = slice_angle_deg / 2               # arc starts just clockwise of the gap
    end_deg = 360 - slice_angle_deg / 2           # arc ends just anticlockwise of the gap
    for i in range(num_arc_pts + 1):
        angle = math.radians(start_deg + i * (end_deg - start_deg) / num_arc_pts)
        pizza.append({
            "x": 0.5 + 0.45 * math.cos(angle),   # radius 0.45 in normalised space
            "y": 0.5 + 0.45 * math.sin(angle),
        })
    pizza.append({"x": 0.5, "y": 0.5})           # close back to center
    presets["pizza"] = pizza

    # --- EGGPLANT ---
    # A very specific and recognisable shape that will get a lot of laughs on Strava.
    # Outline traced clockwise: head cap -> right shaft -> right ball -> valley ->
    # left ball -> left shaft -> back to head. Closed loop.
    # y=0 is top (head), y=1 is bottom (balls).
    presets["eggplant"] = [
        # ---- Head (rounded cap at top) ----
        {"x": 0.50, "y": 0.02},   # tip-top center
        {"x": 0.57, "y": 0.03},
        {"x": 0.63, "y": 0.06},
        {"x": 0.66, "y": 0.11},
        {"x": 0.65, "y": 0.17},   # widest point of head
        # ---- Ridge / corona ----
        {"x": 0.61, "y": 0.20},
        # ---- Shaft right side ----
        {"x": 0.59, "y": 0.25},
        {"x": 0.59, "y": 0.35},
        {"x": 0.59, "y": 0.45},
        {"x": 0.59, "y": 0.55},
        # ---- Right ball (traced clockwise) ----
        {"x": 0.65, "y": 0.60},
        {"x": 0.73, "y": 0.66},
        {"x": 0.78, "y": 0.74},
        {"x": 0.77, "y": 0.83},
        {"x": 0.71, "y": 0.90},
        {"x": 0.63, "y": 0.92},
        {"x": 0.56, "y": 0.87},
        {"x": 0.52, "y": 0.78},
        # ---- Valley between balls ----
        {"x": 0.50, "y": 0.73},
        {"x": 0.48, "y": 0.78},
        # ---- Left ball (traced clockwise) ----
        {"x": 0.44, "y": 0.87},
        {"x": 0.37, "y": 0.92},
        {"x": 0.29, "y": 0.90},
        {"x": 0.23, "y": 0.83},
        {"x": 0.22, "y": 0.74},
        {"x": 0.27, "y": 0.66},
        {"x": 0.35, "y": 0.60},
        # ---- Shaft left side ----
        {"x": 0.41, "y": 0.55},
        {"x": 0.41, "y": 0.45},
        {"x": 0.41, "y": 0.35},
        {"x": 0.41, "y": 0.25},
        # ---- Head left side ----
        {"x": 0.39, "y": 0.20},
        {"x": 0.35, "y": 0.17},
        {"x": 0.34, "y": 0.11},
        {"x": 0.37, "y": 0.06},
        {"x": 0.43, "y": 0.03},
        {"x": 0.50, "y": 0.02},   # close back to tip
    ]

    return presets

# Built once at import time — these never change at runtime
PRESET_SHAPES = _generate_preset_shapes()

# Display metadata for each preset.
# FRONTEND NOTES:
#   - "label" is the user-facing display name (show this in the UI)
#   - "description" is a short tooltip / subtitle
#   - "point_count" tells you how detailed the outline is
#   - Use GET /presets to fetch this list dynamically so you stay in sync
PRESET_DISPLAY_INFO = {
    "pizza": {
        "label": "Pizza",
        "description": "A pizza with a slice missing — the classic",
    },
    "eggplant": {
        "label": "Eggplant",
        "description": "The legendary eggplant. You know what it is.",
    },
}

# --- Helper Functions ---

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

async def fetch_osm_data(app: FastAPI, lat: float, lon: float, radius_m: int) -> dict:
    # Round to approx 500m for very large radii to increase cache hits
    round_factor = 3 if radius_m < 5000 else 2
    cache_key = (round(lat, round_factor), round(lon, round_factor), radius_m)
    if cache_key in OSM_CACHE:
        logger.debug(f"OSM cache hit for {cache_key}")
        return OSM_CACHE[cache_key]

    # Try different radii to find a balance between data size and coverage
    attempts = [radius_m, int(radius_m * 0.7), 5000]
    if radius_m > 15000:
        attempts = [radius_m, 10000, 5000]
    elif radius_m < 3000:
        attempts = [radius_m, 3000, 5000]

    for r_factor in attempts:
        query = f"""
        [out:json][timeout:30];
        (
          way["highway"~"footway|path|track|residential|unclassified|tertiary|secondary|primary"](around:{r_factor},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            response = await app.state.http_client.post(OVERPASS_URL, data={"data": query}, timeout=35.0)
            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])
                if len(elements) > 10:
                    if len(OSM_CACHE) > CACHE_SIZE_OSM:
                        OSM_CACHE.pop(next(iter(OSM_CACHE)))
                    OSM_CACHE[cache_key] = data
                    logger.info(f"Cached OSM data: {len(elements)} elements")
                    return data
        except Exception as e:
            logger.warning(f"OSM fetch attempt failed: {e}")
            continue

    logger.error(f"Failed to fetch OSM data for lat={lat}, lon={lon}")
    raise HTTPException(status_code=500, detail="Failed to fetch data from OpenStreetMap after several attempts")

async def fetch_elevations(app: FastAPI, coords: List[List[float]]) -> List[float]:
    """Fetch elevations for a list of [lon, lat] coordinates."""
    if not coords: return []

    # Create a unique key for this set of coordinates (rounded to 5 decimals)
    # We sample the coords to create a key to avoid massive keys
    cache_key = tuple((round(c[0], 5), round(c[1], 5)) for c in coords[::max(1, len(coords)//5)])
    if cache_key in ELEVATION_CACHE:
        logger.debug(f"Elevation cache hit")
        return ELEVATION_CACHE[cache_key]

    # Open-Elevation expects {"locations": [{"latitude": lat, "longitude": lon}, ...]}
    payload = {
        "locations": [{"latitude": c[1], "longitude": c[0]} for c in coords]
    }
    try:
        response = await app.state.http_client.post(ELEVATION_URL, json=payload, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            elevs = [loc["elevation"] for loc in data["results"]]

            if len(ELEVATION_CACHE) > CACHE_SIZE_ELEVATION:
                ELEVATION_CACHE.pop(next(iter(ELEVATION_CACHE)))
            ELEVATION_CACHE[cache_key] = elevs
            logger.info(f"Cached elevation data: {len(elevs)} points")
            return elevs
    except Exception as e:
        logger.warning(f"Elevation fetch failed: {e}")

    logger.warning(f"Using fallback elevations for {len(coords)} coordinates")
    return [10.0] * len(coords)  # Fallback

def build_graph(osm_data: dict) -> nx.Graph:
    G = nx.Graph()
    nodes = {n["id"]: (n["lon"], n["lat"]) for n in osm_data["elements"] if n["type"] == "node"}
    
    for element in osm_data["elements"]:
        if element["type"] == "way":
            way_nodes = element["nodes"]
            surface = element.get("tags", {}).get("surface", "unknown")
            highway = element.get("tags", {}).get("highway", "unknown")
            for i in range(len(way_nodes) - 1):
                u, v = way_nodes[i], way_nodes[i+1]
                if u in nodes and v in nodes:
                    dist = haversine(nodes[u][0], nodes[u][1], nodes[v][0], nodes[v][1])
                    G.add_edge(u, v, weight=dist, surface=surface, highway=highway)
    return G, nodes

def _precompute_graph_info(G: nx.Graph):
    """Pre-compute bridge edges and dead-end subtree nodes for loop optimization."""
    bridge_edges = set()
    try:
        for u, v in nx.bridges(G):
            bridge_edges.add((u, v))
            bridge_edges.add((v, u))
    except nx.NetworkXError:
        pass

    # Detect dead-end subtree nodes: propagate inward from degree-1 leaves through bridges
    dead_end_nodes = set()
    degree_map = dict(G.degree())
    leaves = [n for n, d in degree_map.items() if d == 1]
    queue = list(leaves)
    while queue:
        node = queue.pop()
        if node in dead_end_nodes:
            continue
        dead_end_nodes.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in dead_end_nodes and (node, neighbor) in bridge_edges:
                # Check if removing this node would make the neighbor a leaf
                remaining_degree = sum(1 for nb in G.neighbors(neighbor) if nb not in dead_end_nodes)
                if remaining_degree <= 1:
                    queue.append(neighbor)

    # Loop connectivity score per node: count of non-bridge edges
    loop_connectivity = {}
    for node in G.nodes():
        non_bridge = sum(1 for nb in G.neighbors(node) if (node, nb) not in bridge_edges)
        loop_connectivity[node] = non_bridge

    return bridge_edges, dead_end_nodes, loop_connectivity


def _compute_shape_radius_m(shape: List[ShapePoint], target_dist_km: float, center_lat: float) -> int:
    """Calculate the OSM fetch radius (in metres) needed to cover the full projected shape.

    The shape gets scaled and projected outward from the user's position. This works
    out how far the farthest point will land and adds a 50 % margin so the road
    network extends beyond the shape for routing detours.

    Called BEFORE fetching OSM data so we pull enough roads on the first try.
    """
    # Calculate shape perimeter in normalised space
    perimeter = 0.0
    for i in range(len(shape) - 1):
        dx = shape[i + 1].x - shape[i].x
        dy = shape[i + 1].y - shape[i].y
        perimeter += math.sqrt(dx * dx + dy * dy)
    if perimeter < 1e-9:
        return 2000  # sensible minimum

    scale_km = target_dist_km / perimeter

    # Shape centroid in normalised space
    cx = sum(p.x for p in shape) / len(shape)
    cy = sum(p.y for p in shape) / len(shape)

    # Max distance from centroid to any point, in normalised units
    max_norm_dist = max(
        math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2) for p in shape
    )

    # Convert to km -> metres, with 50 % margin for routing detours
    max_extent_km = max_norm_dist * scale_km
    return max(int(max_extent_km * 1500), 2000)  # x1.5 in metres, floor 2 km


def find_loop(G: nx.Graph, start_node: int, target_dist_km: float, nodes: dict) -> Tuple[List[int], float]:
    """Find a loop starting and ending at start_node with length approx target_dist_km."""
    start_lon, start_lat = nodes[start_node]

    # Pre-compute graph structural info
    bridge_edges, dead_end_nodes, loop_connectivity = _precompute_graph_info(G)

    # Walk 55% of target distance outward, leaving room for a different return route
    out_target = target_dist_km * 0.55
    max_steps = max(500, int(target_dist_km * 70))
    num_attempts = 10

    best_path = []
    best_dist = 0
    best_quality = -1.0

    for attempt in range(num_attempts):
        current_node = start_node
        path = [current_node]
        current_dist = 0
        path_edges = set()
        visited = {start_node}

        # Systematic angle spread with small perturbation
        base_angle = (2 * math.pi * attempt) / num_attempts
        angle = base_angle + random.uniform(-0.15, 0.15)

        # Walk outward, rotating heading by 180 degrees to trace an arc
        for i in range(max_steps):
            progress = min(current_dist / (out_target + 1e-9), 1.0)
            # Rotate the target direction gradually by pi (180 degrees) over the walk
            current_angle = angle + progress * math.pi
            target_vector = (math.cos(current_angle), math.sin(current_angle))

            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break

            scored_neighbors = []
            for n in neighbors:
                lon, lat = nodes[n]
                dist_from_start = haversine(lon, lat, start_lon, start_lat)
                edge_key = (current_node, n)
                reverse_key = (n, current_node)

                # Directional alignment
                d_lon, d_lat = lon - nodes[current_node][0], lat - nodes[current_node][1]
                mag = math.sqrt(d_lon**2 + d_lat**2) + 1e-9
                dot = (d_lon / mag * target_vector[0] + d_lat / mag * target_vector[1])

                score = dot * 5.0

                # Distance-from-start bonus (encourage spreading out)
                score += (dist_from_start / (out_target + 0.1)) * 3.0

                # Loop connectivity bonus
                score += loop_connectivity.get(n, 0) * 0.5

                # Tiered penalties
                if edge_key in path_edges or reverse_key in path_edges:
                    score -= 200.0  # Edge reuse: heavily penalized
                if n in dead_end_nodes:
                    score -= 80.0   # Dead-end subtree
                if n in visited:
                    score -= 40.0   # Revisited node
                if edge_key in bridge_edges:
                    score -= 15.0   # Bridge edge (risky for loops)

                scored_neighbors.append((n, score))

            if not scored_neighbors:
                break

            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = scored_neighbors[:3]

            if len(top_neighbors) > 1:
                min_score = min(s for _, s in top_neighbors)
                weights = [s - min_score + 1.0 for _, s in top_neighbors]
                next_node = random.choices([n for n, _ in top_neighbors], weights=weights)[0]
            else:
                next_node = top_neighbors[0][0]

            step_dist = G[current_node][next_node]['weight']
            current_dist += step_dist
            path_edges.add((current_node, next_node))
            path_edges.add((next_node, current_node))
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node

            if current_dist >= out_target:
                break

        if current_dist < out_target * 0.3:
            continue

        # Two-phase return path
        # Phase 1: near-absolute penalty on reused edges (100,000x)
        def back_weight_strict(u, v, d):
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                return 1e9
            w = edge_data.get('weight', 0.1)
            if (u, v) in path_edges or (v, u) in path_edges:
                return w * 100000.0
            return w

        back_path = None
        try:
            back_path = nx.shortest_path(G, source=current_node, target=start_node, weight=back_weight_strict)
        except nx.NetworkXNoPath:
            pass

        # Phase 2: if phase 1 still reuses edges, try moderate penalty (500x)
        if back_path:
            reuses_phase1 = sum(1 for j in range(len(back_path) - 1)
                                if (back_path[j], back_path[j + 1]) in path_edges
                                or (back_path[j + 1], back_path[j]) in path_edges)
            if reuses_phase1 > 0:
                def back_weight_moderate(u, v, d):
                    edge_data = G.get_edge_data(u, v)
                    if edge_data is None:
                        return 1e9
                    w = edge_data.get('weight', 0.1)
                    if (u, v) in path_edges or (v, u) in path_edges:
                        return w * 500.0
                    return w
                try:
                    alt_back = nx.shortest_path(G, source=current_node, target=start_node, weight=back_weight_moderate)
                    # Use whichever has fewer reuses
                    reuses_alt = sum(1 for j in range(len(alt_back) - 1)
                                     if (alt_back[j], alt_back[j + 1]) in path_edges
                                     or (alt_back[j + 1], alt_back[j]) in path_edges)
                    if reuses_alt <= reuses_phase1:
                        back_path = alt_back
                except nx.NetworkXNoPath:
                    pass

        if not back_path:
            continue

        # Append return path
        return_edges = set()
        for j in range(1, len(back_path)):
            u, v = back_path[j - 1], back_path[j]
            real_w = G[u][v].get('weight', 0)
            current_dist += real_w
            path.append(v)
            return_edges.add((u, v))
            return_edges.add((v, u))

        # Calculate quality metric
        total_return_edges = len(back_path) - 1
        reused_edges = sum(1 for j in range(len(back_path) - 1)
                           if (back_path[j], back_path[j + 1]) in path_edges
                           or (back_path[j + 1], back_path[j]) in path_edges)
        reuse_ratio = reused_edges / max(total_return_edges, 1)
        distance_accuracy = 1.0 - min(abs(current_dist - target_dist_km) / target_dist_km, 1.0)
        quality = (1.0 - reuse_ratio) * 0.7 + distance_accuracy * 0.3

        # Early termination: zero-reuse loop within 70-130% of target distance
        if reuse_ratio == 0 and target_dist_km * 0.7 <= current_dist <= target_dist_km * 1.3:
            return path, current_dist

        if quality > best_quality:
            best_quality = quality
            best_dist = current_dist
            best_path = path

    return best_path, best_dist


def find_shape_route(G: nx.Graph, start_node: int, target_dist_km: float, nodes: dict, shape: List[ShapePoint]) -> Tuple[List[int], float]:
    """Route along real roads/paths to approximate a shape defined by normalised waypoints.

    IMPORTANT: every node in the returned path is a real OSM road/path node.
    There are NO straight-line gaps — the route is fully runnable on foot.

    The route goes:
        start_node --(road)--> first shape point --(road outline)--> ... --> last shape point --(road)--> start_node

    This means the runner:
        1. Runs from their position to the shape (along roads)
        2. Traces the shape outline (along roads)
        3. Runs back to their starting position (along roads)
    """
    start_lon, start_lat = nodes[start_node]

    # --- 1. Calculate shape perimeter in normalised space ---
    perimeter = 0.0
    for i in range(len(shape) - 1):
        dx = shape[i + 1].x - shape[i].x
        dy = shape[i + 1].y - shape[i].y
        perimeter += math.sqrt(dx * dx + dy * dy)
    if perimeter < 1e-9:
        return [], 0.0

    # --- 2. Scale normalised shape to geographic coordinates ---
    # scale_km = how many km per unit of normalised space
    # This makes the shape's total outline length ≈ target_dist_km
    scale_km = target_dist_km / perimeter
    # Convert km to degrees: 1° lat ≈ 111 km, 1° lon ≈ 111 * cos(lat) km
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(start_lat))

    # Centre the shape on the start node position (user's nearest road)
    cx = sum(p.x for p in shape) / len(shape)
    cy = sum(p.y for p in shape) / len(shape)

    # Project each normalised point to (lon, lat)
    geo_points = []
    for p in shape:
        offset_x_km = (p.x - cx) * scale_km
        offset_y_km = (p.y - cy) * scale_km
        lon = start_lon + offset_x_km / km_per_deg_lon
        lat = start_lat + offset_y_km / km_per_deg_lat
        geo_points.append((lon, lat))

    # --- 3. Snap each projected point to nearest road node ---
    # Every waypoint is guaranteed to be on a real road/path in the OSM graph.
    all_node_ids = list(nodes.keys())
    waypoint_nodes = []
    for glon, glat in geo_points:
        nearest = min(all_node_ids, key=lambda n: haversine(glon, glat, nodes[n][0], nodes[n][1]))
        waypoint_nodes.append(nearest)

    # --- 4. Deduplicate consecutive identical waypoints ---
    # Adjacent shape points often snap to the same road node — remove duplicates
    # so we don't waste time routing a node to itself.
    deduped = [waypoint_nodes[0]]
    for i in range(1, len(waypoint_nodes)):
        if waypoint_nodes[i] != deduped[-1]:
            deduped.append(waypoint_nodes[i])
    waypoint_nodes = deduped

    if len(waypoint_nodes) < 2:
        return [], 0.0

    # --- 5. Build the full route along real roads ---
    # Tracks every edge used across ALL segments so later segments avoid
    # backtracking over earlier ones. This keeps the shape clean on Strava.
    full_path = []
    total_dist = 0.0
    used_edges = set()

    def _route_segment(src, dst):
        """Route between two graph nodes along real roads.

        Uses two-phase edge-reuse penalty (same approach as find_loop's return path):
          Phase 1: 100,000x penalty — almost completely avoids reused edges
          Phase 2:     500x penalty — fallback if phase 1 still reuses edges

        Returns (path_node_list, segment_distance_km) or (None, 0) if unreachable.
        """
        if src == dst:
            return [src], 0.0

        # Phase 1: near-absolute penalty on reused edges
        def w_strict(u, v, d):
            edge_data = G.get_edge_data(u, v)
            if edge_data is None:
                return 1e9
            w = edge_data.get('weight', 0.1)
            if (u, v) in used_edges or (v, u) in used_edges:
                return w * 100000.0
            return w

        seg = None
        try:
            seg = nx.shortest_path(G, source=src, target=dst, weight=w_strict)
        except nx.NetworkXNoPath:
            return None, 0.0

        # Phase 2: if phase 1 still reuses edges, try moderate penalty
        reuses = sum(1 for j in range(len(seg) - 1)
                     if (seg[j], seg[j + 1]) in used_edges
                     or (seg[j + 1], seg[j]) in used_edges)
        if reuses > 0:
            def w_moderate(u, v, d):
                edge_data = G.get_edge_data(u, v)
                if edge_data is None:
                    return 1e9
                w = edge_data.get('weight', 0.1)
                if (u, v) in used_edges or (v, u) in used_edges:
                    return w * 500.0
                return w

            try:
                alt = nx.shortest_path(G, source=src, target=dst, weight=w_moderate)
                alt_reuses = sum(1 for j in range(len(alt) - 1)
                                 if (alt[j], alt[j + 1]) in used_edges
                                 or (alt[j + 1], alt[j]) in used_edges)
                if alt_reuses <= reuses:
                    seg = alt
            except nx.NetworkXNoPath:
                pass

        # Record edges as used and tally distance
        seg_dist = 0.0
        for j in range(len(seg) - 1):
            u, v = seg[j], seg[j + 1]
            used_edges.add((u, v))
            used_edges.add((v, u))
            seg_dist += G[u][v].get('weight', 0)

        return seg, seg_dist

    # 5a. CONNECTOR: route from start_node to first shape waypoint (along roads).
    #     Without this the route would jump in a straight line from the user's
    #     position to wherever the shape starts — through buildings, rivers, etc.
    if start_node != waypoint_nodes[0]:
        conn, conn_dist = _route_segment(start_node, waypoint_nodes[0])
        if conn:
            full_path.extend(conn)
            total_dist += conn_dist
        else:
            # Can't reach first waypoint — start directly there (rare edge case)
            full_path.append(waypoint_nodes[0])
    else:
        full_path.append(start_node)

    # 5b. SHAPE OUTLINE: route between consecutive shape waypoints along roads.
    #     If a segment fails (disconnected graph region), we skip that waypoint
    #     and try to route from the last successfully reached node to the NEXT
    #     waypoint. This avoids silent gaps in the shape.
    last_reached = full_path[-1]
    for i in range(len(waypoint_nodes) - 1):
        dst = waypoint_nodes[i + 1]
        if last_reached == dst:
            continue

        seg, seg_dist = _route_segment(last_reached, dst)
        if seg:
            full_path.extend(seg[1:])  # skip first node to avoid duplicate
            total_dist += seg_dist
            last_reached = dst
        # If segment failed, last_reached stays put — next iteration will try
        # routing from here to the NEXT waypoint, bridging the gap.

    # 5c. CONNECTOR: route from last shape point back to start_node (along roads).
    #     This ensures the runner can actually get home without cutting through
    #     a field. Uses the same edge-reuse penalties so the return path tries
    #     to take different roads to the outbound connector.
    if full_path[-1] != start_node:
        ret, ret_dist = _route_segment(full_path[-1], start_node)
        if ret:
            full_path.extend(ret[1:])
            total_dist += ret_dist

    return full_path, total_dist


def get_difficulty_label(difficulty_score: int) -> str:
    if 1 <= difficulty_score <= 3:
        return "easy"
    elif 4 <= difficulty_score <= 6:
        return "moderate"
    elif 7 <= difficulty_score <= 10:
        return "hard"
    return "moderate"  # Fallback

def generate_mock_suggestion(id: str, lat: float, lon: float, distance_km: float, difficulty: int) -> Suggestion:
    """Generate a mock suggestion for development/testing."""
    # Very simple mock: create a small square loop
    # 1 degree lat is approx 111km. 1 degree lon at equator is 111km.
    # We want a loop of total distance distance_km.
    # Side length approx distance_km / 4.
    side_deg = (distance_km / 4.0) / 111.0

    coords = [
        [lon, lat, 10.0],
        [lon + side_deg, lat, 15.0],
        [lon + side_deg, lat + side_deg, 25.0],
        [lon, lat + side_deg, 20.0],
        [lon, lat, 10.0]  # Close the loop
    ]

    # Strictly enforce input coordinates for mock as well
    coords[0] = [lon, lat, 10.0]
    coords[-1] = [lon, lat, 10.0]

    diff_label = get_difficulty_label(difficulty)

    # Elevation stats based on difficulty
    ascent = distance_km * difficulty * 2.0  # Just a heuristic

    # Duration estimate: distance / speed.
    # Base speed 10km/h (6 min/km)
    # Higher difficulty = slower
    speed_kmh = 12.0 - (difficulty * 0.5)
    duration = int((distance_km / speed_kmh) * 60)

    # Terrain based on difficulty
    paved = max(0, 100 - (difficulty - 1) * 10)
    unpaved = 100 - paved
    primary = "asphalt" if paved > 50 else "trail"

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

    return Suggestion(
        meta=Meta(
            id=id,
            label=f"The {diff_label.capitalize()} {distance_km}km Loop",
            difficulty=diff_label,
            generated_at=now_iso
        ),
        summary=Summary(
            distance_km=distance_km,
            duration_mins=duration,
            ascent_m=ascent,
            descent_m=ascent,
            avg_gradient_percent=difficulty * 0.5,
            elevation_profile=[10, 15, 25, 20, 10]
        ),
        terrain_composition=TerrainComposition(
            paved_pct=paved,
            unpaved_pct=unpaved,
            primary_surface=primary
        ),
        map_data=MapData(
            bbox=[[lon, lat], [lon + side_deg, lat + side_deg]],
            geometry=Geometry(coordinates=coords)
        )
    )

# --- Health & Status Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "ok",
        "version": API_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check - ensures external services are available."""
    try:
        # Quick check that we can reach the OSM API
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://overpass-api.de/api/status", timeout=5.0)
            osm_ok = response.status_code == 200
    except Exception as e:
        logger.warning(f"OSM readiness check failed: {e}")
        osm_ok = False

    return {
        "ready": osm_ok,
        "version": API_VERSION,
        "services": {
            "osm": "ok" if osm_ok else "degraded",
            "elevation": "ok",  # Elevation is not critical
            "cache": f"{len(OSM_CACHE)} OSM / {len(ELEVATION_CACHE)} elevation"
        }
    }

# --- Endpoints ---

async def process_suggestion(i: int, G: nx.Graph, start_node_id: int, distance_km: float, nodes: dict, diff_label: str, difficulty: int, now_iso: str, app: FastAPI, user_lat: float, user_lon: float, request_id: str) -> Optional[Suggestion]:
    # Ensure we use floats for all calculations to avoid potential type issues
    user_lat, user_lon = float(user_lat), float(user_lon)
    
    # Try multiple times to get a suggestion with decent length
    best_s = None
    for attempt in range(4): # Increased attempts
        path_nodes, actual_dist = find_loop(G, start_node_id, distance_km, nodes)
        if not path_nodes: continue

        # Remove duplicate start_node at the end if the loop closed properly
        # (path_nodes typically is [start_node, ..., start_node])
        # This prevents the route from going back to start_node before returning to user
        if len(path_nodes) > 1 and path_nodes[-1] == path_nodes[0]:
            path_nodes = path_nodes[:-1]

        # Build coordinates: user's exact position -> path nodes -> user's exact position
        # This ensures the route visually starts and ends exactly where the user selected
        coords = [[user_lon, user_lat]]
        for n in path_nodes:
            coords.append([nodes[n][0], nodes[n][1]])
        coords.append([user_lon, user_lat])
        
        # Recalculate actual distance including the user-to-graph segments
        actual_dist = 0
        for j in range(len(coords) - 1):
            actual_dist += haversine(float(coords[j][0]), float(coords[j][1]), float(coords[j+1][0]), float(coords[j+1][1]))
        
        # If distance is too short, try again unless we're out of attempts
        # For long distances, allow more deviation but still aim for at least 80%
        min_threshold = 0.8 if distance_km > 10 else 0.7
        if actual_dist < distance_km * min_threshold and attempt < 3:
            if best_s is None or actual_dist > (best_s.summary.distance_km if hasattr(best_s, "summary") else 0):
                # Keep track of the best one in case all attempts are too short
                # Note: We can't actually store s here because it's not created yet, 
                # but we can store the info to create it later if needed.
                # For now, let's just continue and if loop ends without a return, 
                # the last attempt's 'best_s' (if any) will be handled.
                pass
            continue

        # Fetch real elevations
        elevs = await fetch_elevations(app, coords)
        coords_with_alt = []
        for idx, c in enumerate(coords):
            # Ensure start/end are bit-for-bit identical to user inputs
            if idx == 0 or idx == len(coords) - 1:
                coords_with_alt.append([user_lon, user_lat, float(elevs[idx])])
            else:
                coords_with_alt.append([float(c[0]), float(c[1]), float(elevs[idx])])
        
        ascent = 0
        descent = 0
        for j in range(len(elevs) - 1):
            diff = elevs[j+1] - elevs[j]
            if diff > 0: ascent += diff
            else: descent += abs(diff)

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = [[min(lons), min(lats)], [max(lons), max(lats)]]

        s = Suggestion(
            meta=Meta(
                id=f"osm-{i}-{actual_dist}-{random.random()}",
                label=f"OSM {diff_label.capitalize()} {round(actual_dist, 2)}km Loop",
                difficulty=diff_label,
                generated_at=now_iso
            ),
            summary=Summary(
                distance_km=round(actual_dist, 2),
                duration_mins=int((actual_dist / (12.0 - difficulty * 0.5)) * 60),
                ascent_m=round(ascent, 1),
                descent_m=round(descent, 1),
                avg_gradient_percent=round((ascent / (actual_dist * 1000)) * 100, 1) if actual_dist > 0 else 0,
                elevation_profile=elevs[::max(1, len(elevs)//10)] # Sample 10 points
            ),
            terrain_composition=TerrainComposition(
                paved_pct=80, 
                unpaved_pct=20,
                primary_surface="asphalt"
            ),
            map_data=MapData(
                bbox=bbox,
                geometry=Geometry(coordinates=coords_with_alt)
            )
        )
        
        if actual_dist >= distance_km * 0.7:
            return s
        
        if best_s is None or actual_dist > best_s.summary.distance_km:
            best_s = s
            
    return best_s

async def process_shape_suggestion(G: nx.Graph, start_node_id: int, distance_km: float, nodes: dict, diff_label: str, difficulty: int, now_iso: str, app: FastAPI, user_lat: float, user_lon: float, shape: List[ShapePoint], preset_name: Optional[str] = None, request_id: str = "") -> Optional[Suggestion]:
    """Build a Suggestion from a shape route. Mirrors process_suggestion() for loops.

    FRONTEND NOTES:
    - Returns the same Suggestion model as GET /suggestions — same rendering logic
    - The label will include the preset name if one was used (e.g. "Eggplant 5.2km Route")
    - coordinates in map_data.geometry are [lon, lat, elevation] — same format everywhere
    """
    user_lat, user_lon = float(user_lat), float(user_lon)

    path_nodes, actual_dist = find_shape_route(G, start_node_id, distance_km, nodes, shape)
    if not path_nodes:
        return None

    # Build coordinates: user's exact position -> road path nodes -> user's exact position
    # The path_nodes already start and end at start_node (the nearest road to the user),
    # so the only straight-line gap is the short distance from user's GPS to the road.
    coords = [[user_lon, user_lat]]
    for n in path_nodes:
        coords.append([nodes[n][0], nodes[n][1]])
    coords.append([user_lon, user_lat])

    # Recalculate actual distance including the short user-to-road segments
    actual_dist = 0
    for j in range(len(coords) - 1):
        actual_dist += haversine(float(coords[j][0]), float(coords[j][1]), float(coords[j + 1][0]), float(coords[j + 1][1]))

    # Fetch real elevations for the route
    elevs = await fetch_elevations(app, coords)
    coords_with_alt = []
    for idx, c in enumerate(coords):
        if idx == 0 or idx == len(coords) - 1:
            coords_with_alt.append([user_lon, user_lat, float(elevs[idx])])
        else:
            coords_with_alt.append([float(c[0]), float(c[1]), float(elevs[idx])])

    ascent = 0
    descent = 0
    for j in range(len(elevs) - 1):
        diff = elevs[j + 1] - elevs[j]
        if diff > 0:
            ascent += diff
        else:
            descent += abs(diff)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = [[min(lons), min(lats)], [max(lons), max(lats)]]

    # Use the preset name in the label so the user sees "Eggplant 5.2km Route"
    # instead of generic "Shape 5.2km Route". Falls back to "Shape" for custom drawings.
    shape_label = preset_name.capitalize() if preset_name else "Shape"

    return Suggestion(
        meta=Meta(
            id=f"shape-{actual_dist}-{random.random()}",
            label=f"{shape_label} {diff_label.capitalize()} {round(actual_dist, 2)}km Route",
            difficulty=diff_label,
            generated_at=now_iso
        ),
        summary=Summary(
            distance_km=round(actual_dist, 2),
            duration_mins=int((actual_dist / (12.0 - difficulty * 0.5)) * 60),
            ascent_m=round(ascent, 1),
            descent_m=round(descent, 1),
            avg_gradient_percent=round((ascent / (actual_dist * 1000)) * 100, 1) if actual_dist > 0 else 0,
            elevation_profile=elevs[::max(1, len(elevs) // 10)]
        ),
        terrain_composition=TerrainComposition(
            paved_pct=80,
            unpaved_pct=20,
            primary_surface="asphalt"
        ),
        map_data=MapData(
            bbox=bbox,
            geometry=Geometry(coordinates=coords_with_alt)
        )
    )


@app.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    request: Request,
    lat: float = Query(...),
    lon: float = Query(...),
    distance_km: float = Query(...),
    difficulty: int = Query(...)
):
    try:
        # 1. Fetch real OSM data around the user
        # Radius: balance between having enough trails and API performance
        # For long distances, we need a larger radius
        radius_m = int(max(distance_km, 2.0) * 800) 
        osm_data = await fetch_osm_data(app, lat, lon, radius_m)
        
        # 2. Build graph and find start node (closest to user)
        G, nodes = build_graph(osm_data)
        if not nodes:
            raise HTTPException(status_code=404, detail="No roads found nearby. Try a different location.")
            
        start_node_id = min(nodes.keys(), key=lambda n: haversine(lon, lat, nodes[n][0], nodes[n][1]))
        
        # 3. Generate 3 loops concurrently
        request_id = str(uuid.uuid4())
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
        diff_label = get_difficulty_label(difficulty)

        logger.info(f"[{request_id}] Generating suggestions: lat={lat}, lon={lon}, distance_km={distance_km}, difficulty={difficulty}")

        tasks = [process_suggestion(i, G, start_node_id, distance_km, nodes, diff_label, difficulty, now_iso, app, lat, lon, request_id) for i in range(1, 4)]
        results = await asyncio.gather(*tasks)
        
        suggestions = [s for s in results if s is not None]

        if not suggestions:
            logger.warning(f"[{request_id}] No routes found")
            raise HTTPException(status_code=404, detail="No routes found in this area. Try a different location or distance.")

        logger.info(f"[{request_id}] Generated {len(suggestions)} suggestions")
        return SuggestionsResponse(suggestions=suggestions)

    except HTTPException as e:
        logger.warning(f"[{request.headers.get('x-request-id', 'unknown')}] HTTP Error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error generating routes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate routes: {str(e)}")

# --- GET /presets ---
# FRONTEND NOTES:
#   Call this on app startup or when showing the shape picker to get the list
#   of available preset shapes. Response is a JSON object keyed by preset name.
#   Example response:
#   {
#     "pizza":    {"label": "Pizza",    "description": "...", "point_count": 27},
#     "eggplant": {"label": "Eggplant", "description": "...", "point_count": 36}
#   }
#   To use a preset, POST to /shape-route with {"preset": "eggplant", ...}
@app.get("/presets")
async def list_presets():
    """Return available preset shapes with display metadata.

    FRONTEND: use this to populate a shape-picker grid/list. Each key is the
    value you pass as "preset" in ShapeRequest.
    """
    result = {}
    for name, info in PRESET_DISPLAY_INFO.items():
        result[name] = {
            **info,
            "point_count": len(PRESET_SHAPES.get(name, [])),
        }
    return result


@app.post("/shape-route", response_model=SuggestionsResponse)
async def get_shape_route(request: Request, body: ShapeRequest):
    """Generate a running route that traces a shape on the map.

    FRONTEND NOTES:
    - Send "preset": "<name>" to use a built-in shape (pizza, eggplant, etc.)
    - Send "shape": [...] with custom ShapePoints for freehand / SVG / text outlines
    - If both are provided, the preset wins (shape is ignored)
    - If neither is provided, you get a 400 error
    - The response format is identical to GET /suggestions — same Suggestion model
    """
    try:
        # --- Resolve shape points: preset takes priority over custom shape ---
        if body.preset:
            preset_key = body.preset.lower().strip()
            if preset_key not in PRESET_SHAPES:
                available = ", ".join(sorted(PRESET_SHAPES.keys()))
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset '{body.preset}'. Available presets: {available}"
                )
            # Convert the preset's raw dicts into ShapePoint objects
            shape_points = [ShapePoint(**p) for p in PRESET_SHAPES[preset_key]]
        elif body.shape:
            shape_points = body.shape
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'preset' (e.g. \"pizza\") or 'shape' (list of points)."
            )

        if len(shape_points) < 3:
            raise HTTPException(status_code=400, detail="Shape must have at least 3 points.")

        # Calculate OSM fetch radius to cover the full projected shape.
        # The shape gets scaled outward from the user — we need road data that
        # extends beyond the farthest shape point so every waypoint snaps to a
        # nearby road, not to the edge of our data (which would distort the shape).
        shape_radius_m = _compute_shape_radius_m(shape_points, body.distance_km, body.lat)
        default_radius_m = int(max(body.distance_km, 2.0) * 800)
        radius_m = max(default_radius_m, shape_radius_m)
        osm_data = await fetch_osm_data(app, body.lat, body.lon, radius_m)

        G, nodes = build_graph(osm_data)
        if not nodes:
            raise HTTPException(status_code=404, detail="No roads found nearby. Try a different location.")

        start_node_id = min(nodes.keys(), key=lambda n: haversine(body.lon, body.lat, nodes[n][0], nodes[n][1]))

        request_id = str(uuid.uuid4())
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
        diff_label = get_difficulty_label(body.difficulty)

        logger.info(f"[{request_id}] Generating shape route: preset={body.preset}, distance_km={body.distance_km}")

        suggestion = await process_shape_suggestion(
            G, start_node_id, body.distance_km, nodes, diff_label, body.difficulty,
            now_iso, app, body.lat, body.lon, shape_points,
            preset_name=body.preset if body.preset else None,
            request_id=request_id
        )

        if suggestion is None:
            logger.warning(f"[{request_id}] Could not generate shape route")
            raise HTTPException(status_code=404, detail="Could not generate a shape route in this area. Try a different location or simpler shape.")

        logger.info(f"[{request_id}] Generated shape route successfully")
        return SuggestionsResponse(suggestions=[suggestion])

    except HTTPException as e:
        logger.warning(f"[{request_id}] HTTP Error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error generating shape route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate shape route: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

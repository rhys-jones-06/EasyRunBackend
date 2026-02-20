# EasyRun API Specification v1.0.0

## Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://api.easyrun.example.com`

All endpoints return JSON with proper HTTP status codes.

---

## Authentication

Currently, the API is open. In production, add:
- API Key: `Authorization: Bearer YOUR_API_KEY`
- Or JWT: `Authorization: Bearer YOUR_JWT_TOKEN`

---

## Health Checks

### GET /health
Check if the API is running.

**Response**: 200 OK
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-02-20T15:30:45.123456Z"
}
```

### GET /ready
Check if all external services are available and API is ready to handle requests.

**Response**: 200 OK
```json
{
  "ready": true,
  "version": "1.0.0",
  "services": {
    "osm": "ok",
    "elevation": "ok",
    "cache": "5 OSM / 12 elevation"
  }
}
```

**Response**: 503 Service Unavailable (if critical services down)

---

## API Endpoints

### GET /presets
Get available preset shapes for route generation.

**Response**: 200 OK
```json
{
  "pizza": {
    "label": "Pizza",
    "description": "A pizza with a slice missing — the classic",
    "point_count": 27
  },
  "eggplant": {
    "label": "Eggplant",
    "description": "The legendary eggplant. You know what it is.",
    "point_count": 36
  }
}
```

**Frontend Usage**:
```typescript
const presets = await fetch('/presets').then(r => r.json());
// Display preset options in UI
```

---

### GET /suggestions
Generate 3 random loop routes based on user location and preferences.

**Query Parameters**:
| Name | Type | Required | Range | Description |
|------|------|----------|-------|-------------|
| `lat` | float | Yes | -90 to 90 | Starting latitude |
| `lon` | float | Yes | -180 to 180 | Starting longitude |
| `distance_km` | float | Yes | 0.1 to 100 | Target distance in km |
| `difficulty` | int | Yes | 1 to 10 | Difficulty level (1=easy, 10=hard) |

**Example Request**:
```bash
GET /suggestions?lat=51.5074&lon=-0.1278&distance_km=5&difficulty=5
```

**Response**: 200 OK
```json
{
  "suggestions": [
    {
      "meta": {
        "id": "osm-1-5.2-0.8234",
        "label": "OSM Moderate 5.2km Loop",
        "difficulty": "moderate",
        "generated_at": "2024-02-20T15:30:45.123456Z"
      },
      "summary": {
        "distance_km": 5.2,
        "duration_mins": 38,
        "ascent_m": 120.5,
        "descent_m": 120.5,
        "avg_gradient_percent": 2.3,
        "elevation_profile": [10, 25, 45, 65, 80, 65, 45, 25, 10]
      },
      "terrain_composition": {
        "paved_pct": 80,
        "unpaved_pct": 20,
        "primary_surface": "asphalt"
      },
      "map_data": {
        "bbox": [
          [-0.128, 51.507],
          [-0.125, 51.510]
        ],
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-0.1278, 51.5074, 10.0],
            [-0.1265, 51.5080, 25.0],
            [-0.1250, 51.5090, 45.0],
            [-0.1278, 51.5074, 10.0]
          ]
        }
      }
    }
    // ... 2 more suggestions
  ]
}
```

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 400 | ValidationError | Invalid query parameters |
| 404 | NotFound | No roads found nearby |
| 500 | ServerError | OSM/elevation API failure |

**Frontend Usage**:
```typescript
async function loadSuggestions(lat: number, lon: number, distance: number, difficulty: number) {
  try {
    const response = await fetch(
      `/suggestions?lat=${lat}&lon=${lon}&distance_km=${distance}&difficulty=${difficulty}`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    // data.suggestions is a Suggestion[]
    return data.suggestions;
  } catch (error) {
    console.error('Failed to load suggestions:', error);
    return [];
  }
}
```

---

### POST /shape-route
Generate a single route that traces a custom shape or preset pattern.

**Request Body** (JSON):
```json
{
  "lat": 51.5074,
  "lon": -0.1278,
  "distance_km": 5,
  "difficulty": 5,
  "preset": "pizza"
}
```

Or with custom shape:
```json
{
  "lat": 51.5074,
  "lon": -0.1278,
  "distance_km": 5,
  "difficulty": 5,
  "shape": [
    {"x": 0.5, "y": 0.0},
    {"x": 1.0, "y": 0.5},
    {"x": 0.5, "y": 1.0},
    {"x": 0.0, "y": 0.5},
    {"x": 0.5, "y": 0.0}
  ]
}
```

**Request Field Descriptions**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lat` | float | Yes | User's latitude (-90 to 90) |
| `lon` | float | Yes | User's longitude (-180 to 180) |
| `distance_km` | float | Yes | Target route distance (0.1-100) |
| `difficulty` | int | Yes | Difficulty level (1-10) |
| `preset` | string | No | Preset name: "pizza", "eggplant" |
| `shape` | array | No | Custom shape points (ignored if preset set) |

**Shape Point Format**:
```typescript
interface ShapePoint {
  x: number; // 0.0 to 1.0 (0=left, 1=right)
  y: number; // 0.0 to 1.0 (0=top, 1=bottom)
}
```

**Response**: 200 OK
```json
{
  "suggestions": [
    {
      "meta": {
        "id": "shape-5.2-0.8234",
        "label": "Pizza Moderate 5.2km Route",
        "difficulty": "moderate",
        "generated_at": "2024-02-20T15:30:45.123456Z"
      },
      "summary": {
        "distance_km": 5.2,
        "duration_mins": 38,
        "ascent_m": 45.0,
        "descent_m": 45.0,
        "avg_gradient_percent": 0.9,
        "elevation_profile": [10, 20, 30, 25, 10]
      },
      "terrain_composition": {
        "paved_pct": 75,
        "unpaved_pct": 25,
        "primary_surface": "asphalt"
      },
      "map_data": {
        "bbox": [[-0.133, 51.503], [-0.122, 51.512]],
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-0.1278, 51.5074, 10.0],
            [-0.1270, 51.5080, 15.0],
            ...
          ]
        }
      }
    }
  ]
}
```

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 400 | ValidationError | Invalid request body or parameters |
| 404 | NotFound | Unknown preset or no roads found |
| 500 | ServerError | Route generation failed |

**Frontend Usage**:
```typescript
async function generateShapeRoute(
  lat: number,
  lon: number,
  distance: number,
  difficulty: number,
  preset: string
) {
  const response = await fetch('/shape-route', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      lat, lon, distance_km: distance, difficulty, preset
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  return data.suggestions[0]; // Single route returned
}
```

---

## Data Models

### Suggestion
```typescript
interface Suggestion {
  meta: Meta;
  summary: Summary;
  terrain_composition?: TerrainComposition;
  map_data: MapData;
}
```

### Meta
```typescript
interface Meta {
  id: string; // Unique identifier
  label: string; // Display name
  difficulty: "easy" | "moderate" | "hard";
  generated_at: string; // ISO8601 timestamp
}
```

### Summary
```typescript
interface Summary {
  distance_km: number; // Total distance
  duration_mins: number; // Estimated time
  ascent_m: number; // Elevation gain
  descent_m: number; // Elevation loss
  avg_gradient_percent: number; // Average slope
  elevation_profile: number[]; // 10-point elevation sample
}
```

### TerrainComposition
```typescript
interface TerrainComposition {
  paved_pct: number; // 0-100
  unpaved_pct: number; // 0-100
  primary_surface: "asphalt" | "concrete" | "trail" | "gravel";
}
```

### MapData
```typescript
interface MapData {
  bbox: [[number, number], [number, number]]; // Bounding box
  geometry: GeoJSON.LineString;
}

// GeoJSON.LineString
interface LineString {
  type: "LineString";
  coordinates: Array<[lon: number, lat: number, elevation: number]>;
}
```

---

## Error Response Format

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Errors**:

### 400 Bad Request
```json
{
  "detail": "Provide either 'preset' (e.g. \"pizza\") or 'shape' (list of points)."
}
```

### 404 Not Found
```json
{
  "detail": "No roads found nearby. Try a different location."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to generate routes: Connection timeout"
}
```

---

## Rate Limiting & Throttling

In production, implement rate limiting:
- **Free tier**: 10 requests/minute per IP
- **Premium**: 100 requests/minute per API key

Use headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1708436400
```

---

## Versioning Strategy

API versions are indicated in the URL path:
- Current: `/api/v1/suggestions`
- Future: `/api/v2/suggestions`

The root path (`/suggestions`) always uses the latest version.

To explicitly use a version:
```bash
GET /api/v1/suggestions?lat=51.5&lon=-0.13&distance_km=5&difficulty=5
```

---

## Performance Notes

- **Request timeout**: 30 seconds
- **Cache**: In-memory (50 OSM, 100 elevation queries)
- **Expected response time**: 2-5 seconds
- **Pagination**: Not needed (max 3 results per request)

Future improvements:
- Redis distributed cache
- Database for user preferences
- Async task queue for heavy computations

---

## Frontend Integration Checklist

- [ ] Generate TypeScript types from OpenAPI spec
- [ ] Implement error handling for all status codes
- [ ] Add retry logic with exponential backoff
- [ ] Cancel pending requests on component unmount
- [ ] Cache routes by ID in client
- [ ] Show loading states during requests
- [ ] Handle network timeouts gracefully
- [ ] Add analytics tracking for API errors
- [ ] Display user-friendly error messages

Easy Run App Backend
A Python REST API that generates optimal running routes based on distance, terrain, and elevation difficulty.
Features

Route Generation: Create running routes of a specified distance from any starting point that return to the origin
Terrain-Based Difficulty Scoring: Factor in elevation changes and terrain types to calculate route difficulty
Smart Road Reuse: Avoids redundant paths when alternatives exist, but intelligently allows road reuse when necessary to meet distance targets
Constraint Satisfaction: Solves complex routing constraints to balance distance accuracy with route quality

How It Works
The backend exposes a REST API that accesses a constraint-satisfaction algorithm. Given a starting location, desired distance, and terrain preferences, it calculates an optimized running route that accounts for:

Physical elevation and terrain difficulty
Distance precision (ensures routes meet target distance)
Road network topology (minimizes backtracking)

Use Case
Ideal for runners seeking varied, difficulty-rated routes without manual planning. The algorithm handles the complex spatial and constraint logic, making it easy to explore new running paths.
Tech Stack

Language: Python
API: REST endpoints
Core: Graph-based routing algorithm with constraint satisfaction logic

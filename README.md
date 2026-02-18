# 🏃 Easy Run App Backend

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()

A sophisticated Python REST API that generates optimal running routes based on distance, terrain preferences, and elevation difficulty. Built for runners who want intelligent route planning without the hassle.

## ✨ Features

- **🗺️ Route Generation**: Create running routes of any specified distance from any starting point that intelligently return to the origin
- **⛰️ Terrain-Based Difficulty Scoring**: Factor in real elevation changes and terrain types to calculate accurate route difficulty
- **🔄 Smart Road Reuse**: Avoids redundant paths when alternatives exist, but intelligently allows road reuse when necessary to meet distance targets
- **🧩 Constraint Satisfaction**: Solves complex routing constraints to perfectly balance distance accuracy with route quality
- **📍 Location-Based**: Works with any geographic location as a starting point

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rhys-jones-06/EasyRunBackend.git
cd EasyRunBackend
```

2. Install dependencies:
```bash
# Dependencies to be added as the project develops
# pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 🔧 How It Works

The backend exposes a REST API that leverages a sophisticated constraint-satisfaction algorithm. Given a starting location, desired distance, and terrain preferences, it calculates an optimized running route that accounts for:

- **Physical elevation and terrain difficulty**: Real-world topography affects route planning
- **Distance precision**: Ensures routes accurately meet target distance requirements
- **Road network topology**: Minimizes backtracking and creates natural, flowing routes
- **User preferences**: Adapts to your specific terrain and difficulty requirements

### Algorithm Approach

The system uses a graph-based routing algorithm that models the road network as a weighted graph, where:
- Nodes represent intersections and points of interest
- Edges represent road segments with attributes (distance, elevation change, terrain type)
- The constraint solver finds optimal paths that satisfy multiple objectives simultaneously

## 💡 Use Cases

Perfect for:
- **Runners** seeking varied, difficulty-rated routes without manual planning
- **Training apps** that need automated route generation
- **Fitness platforms** looking to enhance their route discovery features
- **Anyone** exploring new running paths in unfamiliar areas

The algorithm handles the complex spatial and constraint logic, making it effortless to discover new running paths tailored to your needs.

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **API**: REST endpoints
- **Core Algorithm**: Graph-based routing with constraint satisfaction logic
- **Routing**: Spatial analysis and network optimization

## 📋 API Documentation

*Coming soon - API endpoints and usage examples will be documented here as the project develops*

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License.

## 📬 Contact

Project Link: [https://github.com/rhys-jones-06/EasyRunBackend](https://github.com/rhys-jones-06/EasyRunBackend)

## 🔮 Roadmap

- [ ] Implement core routing algorithm
- [ ] Add REST API endpoints
- [ ] Integrate with mapping services
- [ ] Add route caching and optimization
- [ ] Support multiple terrain types
- [ ] Add route sharing capabilities
- [ ] Mobile app integration

---

*Built with ❤️ for the running community*

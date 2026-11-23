# AI-Driven Smart Dairy Supply Chain Optimization — Full Backend & Relational Database

Complete FastAPI + SQLAlchemy 2.0 + Scikit-Learn + VRPTW Route Optimization backend pre-bundled with a seeded 1,500-row relational database (`dairy_supply_chain.db`).

## Quick Start (Local SQLite Mode — Zero Setup)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open interactive Swagger API documentation at `http://localhost:8000/docs`.

## Production Deployment (Docker + PostgreSQL 16)
```bash
docker-compose up --build
```

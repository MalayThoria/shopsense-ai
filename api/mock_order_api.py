"""
Mock Order Lookup API.

A lightweight Flask server that serves order data from orders.json.
Phase 3 agent calls this API to look up order status and details.
Run in a separate terminal: python api/mock_order_api.py
"""

import json
from pathlib import Path

from flask import Flask, jsonify, Response

app = Flask(__name__)

# Load orders into memory at startup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDERS_FILE = PROJECT_ROOT / "data" / "orders.json"
ORDERS: dict = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))


def _add_cors(response: Response) -> Response:
    """Add CORS headers to every response."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def after_request(response: Response) -> Response:
    """
    Attach CORS headers to all responses.

    Args:
        response: The Flask response object.

    Returns:
        Response with CORS headers added.
    """
    return _add_cors(response)


@app.route("/api/orders/<order_id>", methods=["GET"])
def get_order(order_id: str):
    """
    Look up a single order by its ID.

    Args:
        order_id: The order ID string (e.g. ORD-4892).

    Returns:
        JSON order data with 200, or error message with 404.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return jsonify({"error": "Order not found"}), 404
    return jsonify({"order_id": order_id, **order})


@app.route("/api/orders", methods=["GET"])
def get_all_orders():
    """
    Return all orders.

    Returns:
        JSON object with all orders.
    """
    return jsonify(ORDERS)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "orders_loaded": len(ORDERS)})


if __name__ == "__main__":
    print(f"Loaded {len(ORDERS)} orders from {ORDERS_FILE}")
    print("Mock Order API running at http://localhost:5050")
    print("Endpoints:")
    print("  GET /api/orders/<order_id>  — single order lookup")
    print("  GET /api/orders             — all orders")
    print("  GET /health                 — health check")
    app.run(host="0.0.0.0", port=5050, debug=False)
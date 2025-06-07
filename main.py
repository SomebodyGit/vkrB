from flask import Flask, request, jsonify, send_from_directory
import logging
from datetime import datetime
import os
from route_finder import Coordinates, RouteFinder


log_dir = "./logs/logs_server"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

@app.route("/find_route", methods=["POST"])
def find_route():
    logger.info("Получен запрос на поиск маршрута: %s", request.json)
    try:
        data = request.json
        coords = Coordinates(**data)
        start_coords = (coords.start_lat, coords.start_lon)
        target_coords = (coords.target_lat, coords.target_lon)
        route_finder = RouteFinder(start_coords, target_coords)
        result = route_finder.run()
        logger.info("Маршрут успешно найден, длина: %s метров", result["length"])
        return jsonify(result)
    except ValueError as e:
        logger.error("Ошибка ValueError: %s", str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Произошла ошибка: %s", str(e))
        return jsonify({"error": f"Произошла ошибка: {str(e)}"}), 500

@app.route("/")
def root():
    logger.info("Запрошен корневой маршрут, возврат index.html")
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    logger.info("Запуск приложения, лог-файл: %s", log_filename)
    app.run(host="0.0.0.0", port=5001)
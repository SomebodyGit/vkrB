import osmnx as ox
import networkx as nx
import numpy as np
from math import sqrt, log
from geopy.distance import great_circle
from typing import List, Tuple
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import json

logger = logging.getLogger(__name__)

class Coordinates:
    def __init__(self, start_lat, start_lon, target_lat, target_lon):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.target_lat = target_lat
        self.target_lon = target_lon

class RouteFinder:
    def __init__(self, start_coords: Tuple[float, float], target_coords: Tuple[float, float]):
        self.start_coords = start_coords
        self.target_coords = target_coords
        self.G = None
        self.start_node = None
        self.target_node = None
        self.node_to_index = None
        self.index_to_node = None
        self.graph_dict = None
        self.Q = None
        self.path_nodes = None
        self.agent_path_length = 0
        self.is_q_table_loaded = False

    def load_graph(self):
        logger.info("Загрузка графа для координат: %s -> %s", self.start_coords, self.target_coords)
        center_point = (
            (self.start_coords[0] + self.target_coords[0]) / 2,
            (self.start_coords[1] + self.target_coords[1]) / 2
        )
        dist = great_circle(self.start_coords, self.target_coords).meters
        self.G = ox.graph_from_point(center_point, dist=dist * 1.05, network_type='drive')
        largest_component = max(nx.strongly_connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest_component).copy()

    def geodesic_distance(self, node1, node2):
        coord1 = (self.G.nodes[node1]['y'], self.G.nodes[node1]['x'])
        coord2 = (self.G.nodes[node2]['y'], self.G.nodes[node2]['x'])
        return great_circle(coord1, coord2).kilometers

    def save_q_table(self, filename='logs/q_table.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.Q, f)

    def load_q_table(self, filename='logs/q_table.json'):
        with open(filename, 'r') as f:
            self.Q = json.load(f)

    def is_q_table_compatible(self):
        if self.Q is None:
            return False
        
        for node in self.graph_dict:
            if str(node) not in self.Q:
                return False
            
            if set(self.Q[str(node)].keys()) != set(map(str, self.graph_dict[node])):
                return False
        self.is_q_table_loaded = True
        logger.info("Q-таблица загружена из файла")
        return True

    def compute_params(self, dist):
        if dist > 0 and dist <= 10:
            num_episodes = 2000
            discount_factor = 0.85
            epsilon = 0.05
        elif dist > 10 and dist <= 20:
            num_episodes = 6000
            discount_factor = 0.85
            epsilon = 0.05
        elif dist > 20 and dist <= 30:
            num_episodes = 10000
            discount_factor = 0.89
            epsilon = 0.09
        elif dist > 30 and dist <= 40:
            num_episodes = 25000
            discount_factor = 0.89
            epsilon = 0.09
        elif dist > 40 and dist <= 50:
            num_episodes = 40000
            discount_factor = 0.93
            epsilon = 0.13
        elif dist > 50 and dist <= 60:
            num_episodes = 57000
            discount_factor = 0.93
            epsilon = 0.13
        else:
            num_episodes = 102000
            discount_factor = 0.97
            epsilon = 0.17
        return discount_factor, epsilon, num_episodes

    def prepare_data(self):
        logger.info("Подготовка данных графа")
        self.start_node = ox.distance.nearest_nodes(self.G, self.start_coords[1], self.start_coords[0])
        self.target_node = ox.distance.nearest_nodes(self.G, self.target_coords[1], self.target_coords[0])
        if not nx.has_path(self.G, self.start_node, self.target_node):
            logger.error("Целевой узел недостижим из начального узла")
            raise ValueError("Целевой узел недостижим из начального узла.")
        nodes = list(self.G.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        self.graph_dict = {
            self.node_to_index[node]: [self.node_to_index[neighbor] for neighbor in self.G.neighbors(node)]
            for node in self.G.nodes
        }

    def euclidean_distance(self, node1, node2):
        x1, y1 = self.G.nodes[node1]['x'], self.G.nodes[node1]['y']
        x2, y2 = self.G.nodes[node2]['x'], self.G.nodes[node2]['y']
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def train_agent(self):
        logger.info("Начало обучения агента")
        self.Q = {node: {act: 0.0 for act in self.graph_dict[node]} for node in self.graph_dict}
        self.is_q_table_loaded = False

        dist = self.geodesic_distance(self.start_node, self.target_node)

        learning_rate = 0.13
        discount_factor, epsilon, num_episodes = self.compute_params(dist)
        episode_rewards = []
        for episode in range(num_episodes):
            current_node = self.node_to_index[self.start_node]
            done = False
            steps = 0
            total_reward = 0
            while not done and steps < 10000:
                possible_actions = self.graph_dict[current_node]
                if not possible_actions:
                    break

                if np.random.random() < epsilon:
                    action = np.random.choice(possible_actions)
                else:
                    action = max(possible_actions, key=lambda a: self.Q[current_node][a])
                if action == self.node_to_index[self.target_node]:
                    reward = 100
                    done = True
                else:
                    reward = -self.euclidean_distance(self.index_to_node[current_node], self.target_node)
                total_reward += reward
                next_node = action
                best_next_action = max(self.Q[next_node].values()) if self.Q[next_node] else 0
                self.Q[current_node][action] += learning_rate * (
                    reward + discount_factor * best_next_action - self.Q[current_node][action]
                )
                current_node = next_node
                steps += 1
            episode_rewards.append(total_reward)
        self.save_q_table()
        logger.info("Обучение агента завершено")
        return episode_rewards

    def test_agent(self):
        logger.info("Тестирование агента")
        current_node = self.node_to_index[self.start_node]
        path = [current_node]
        max_steps = 10000000
        steps = 0
        visited = set()
        while current_node != self.node_to_index[self.target_node] and steps < max_steps:
            if current_node in visited:
                break
            visited.add(current_node)
            possible_actions = self.graph_dict[current_node]
            if not possible_actions:
                break
            
            if self.is_q_table_loaded:
                action = max(possible_actions, key=lambda a: self.Q[str(current_node)][str(a)])
            else:
                action = max(possible_actions, key=lambda a: self.Q[current_node][a])
            current_node = action
            path.append(current_node)
            steps += 1
        self.path_nodes = [self.index_to_node[idx] for idx in path]
        self._calculate_path_length()
        success = current_node == self.node_to_index[self.target_node]
        logger.info("Тестирование завершено, маршрут %s", "найден" if success else "не найден")
        return success

    def _calculate_path_length(self):
        self.agent_path_length = 0
        for i in range(len(self.path_nodes) - 1):
            u = self.path_nodes[i]
            v = self.path_nodes[i + 1]
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                if isinstance(edge_data, dict) and 'length' in edge_data:
                    self.agent_path_length += edge_data['length']
                elif all(isinstance(val, dict) for val in edge_data.values()):
                    lengths = [data['length'] for data in edge_data.values() if 'length' in data]
                    if lengths:
                        self.agent_path_length += min(lengths)

    def visualize(self):
        logger.info("Визуализация маршрута")
        path_geometries = []
        for i in range(len(self.path_nodes) - 1):
            u = self.path_nodes[i]
            v = self.path_nodes[i + 1]
            edge_data = self.G.get_edge_data(u, v)
            if edge_data and 'geometry' in edge_data:
                geom = edge_data['geometry']
                coords = [(point[1], point[0]) for point in geom.coords]
                path_geometries.extend(coords[:-1])
            else:
                coords = [
                    (self.G.nodes[u]['y'], self.G.nodes[u]['x']),
                    (self.G.nodes[v]['y'], self.G.nodes[v]['x'])
                ]
                path_geometries.append(coords[0])
        if self.path_nodes:
            last_node = self.path_nodes[-1]
            path_geometries.append((self.G.nodes[last_node]['y'], self.G.nodes[last_node]['x']))
        return path_geometries

    def save_image(self, filename='logs/route.png'):
        fig, ax = ox.plot_graph(self.G, node_size=0, bgcolor='k', show=False, close=False, figsize=(15, 15))
        for i in range(len(self.path_nodes) - 1):
            u = self.path_nodes[i]
            v = self.path_nodes[i + 1]
            edges = self.G.get_edge_data(u, v)
            if edges:
                edge_data = list(edges.values())[0]
                if 'geometry' in edge_data:
                    geom = edge_data['geometry']
                    xs, ys = geom.xy
                    ax.plot(xs, ys, color='blue', linewidth=2)
                else:
                    x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                    x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
                    ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2)
        ax.plot([], [], color='blue', label=f'Путь агента ({self.agent_path_length:.2f} м)')
        ax.legend()
        padding = 0.001
        min_x = min(self.G.nodes[node]['x'] for node in self.path_nodes)
        max_x = max(self.G.nodes[node]['x'] for node in self.path_nodes)
        min_y = min(self.G.nodes[node]['y'] for node in self.path_nodes)
        max_y = max(self.G.nodes[node]['y'] for node in self.path_nodes)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        plt.savefig(filename)
        plt.close()

    def run(self):
        self.load_graph()
        self.prepare_data()
        try:
            self.load_q_table()
            if not self.is_q_table_compatible():
                logger.warning("Загруженная Q-таблица несовместима с текущим графом. Переобучение агента.")
                self.train_agent()
                self.is_q_table_loaded = False
        except FileNotFoundError:
            logger.info("Q-таблица не найдена, обучение агента.")
            self.train_agent()
        success = self.test_agent()
        if not success:
            logger.warning("Путь не найден с загруженной Q-таблицей. Переобучение агента.")
            self.train_agent()
            self.is_q_table_loaded = False
            success = self.test_agent()
        if success:
            path_coords = self.visualize()
            self.save_image('logs/route.png')
            logger.info("Маршрут найден и изображение сохранено.")
            return {"path": path_coords, "length": self.agent_path_length, "image": "route.png"}
        else:
            logger.error("Путь не найден даже после переобучения.")
            raise ValueError("Путь не найден.")
"""
database/neo4j.py — gerenciamento do driver Neo4j.

Inicializado via lifespan do FastAPI, injetado nos endpoints via Depends.
"""

from dotenv import load_dotenv
import os
from neo4j import GraphDatabase, Driver

load_dotenv()

NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "senha123")

_driver: Driver | None = None


def get_driver() -> Driver:
    """Dependency injection para os endpoints FastAPI."""
    if _driver is None:
        raise RuntimeError(
            "Neo4j driver não inicializado. "
            "Verifique se init_driver() foi chamado no lifespan da aplicação."
        )
    return _driver


def init_driver() -> Driver:
    """Cria e verifica a conexão. Chamado no startup da aplicação."""
    global _driver
    _driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _driver.verify_connectivity()
    print("✅ Conexão com Neo4j bem-sucedida!")
    return _driver


def close_driver():
    """Fecha a conexão. Chamado no shutdown da aplicação."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        print("🔌 Conexão com Neo4j encerrada.")
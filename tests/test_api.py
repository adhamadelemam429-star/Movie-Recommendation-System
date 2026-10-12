from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_movies_endpoint():
    response = client.get("/movies/search?q=Inception")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_similar_movies_endpoint():
    response = client.get("/movies/1/similar?top_n=3")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_user_recommendations_endpoint():
    response = client.get("/recommendations/user/1?top_n=3")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

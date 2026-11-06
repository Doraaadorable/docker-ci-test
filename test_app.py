from app import app


def test_home_says_hello_and_accuracy():
    html = app.test_client().get("/").get_data(as_text=True)
    assert "Hello" in html
    assert "accuracy" in html.lower()
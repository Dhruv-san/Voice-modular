from star_ai_app import demo

app = demo.app

@app.get("/health")
def health():
    return {"status": "ok"}

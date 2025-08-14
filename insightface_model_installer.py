from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')  # will download if missing
app.prepare(ctx_id=0)

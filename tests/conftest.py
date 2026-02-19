"""
conftest.py - SHARED SETUP FOR ALL TESTS
=========================================
Pytest automatically loads this file before running any test in the "tests" folder.
We put common things here (sample data, fake Kafka, test client) so every test
file can use them without repeating the same code.
"""

# ----- IMPORTS -----
# asyncio: Python's library for running async code (e.g. "sleep" without blocking).
import asyncio

# patch: Lets us temporarily replace a real class/function with a fake one during tests.
# Example: replace "real Kafka" with "fake Kafka" so tests don't need a real server.
from unittest.mock import patch

# pytest: The testing framework. Provides fixtures, assert helpers, and test discovery.
import pytest

# TestClient: FastAPI's way to call your API from inside a test (no real HTTP server needed).
# You do client.get("/health") and get back a response like a real browser would.
from fastapi.testclient import TestClient


# ----- SAMPLE REQUEST DATA -----
# This is a real example of the JSON body our API expects for POST /predict.
# We use it in tests so we don't have to type it in every test; we just refer to SAMPLE_VIDEO_QC_REQUEST.
# Keys must match what the API expects: request_id, sku_id (list), caption, video_id, video_path.
SAMPLE_VIDEO_QC_REQUEST = {
    "request_id": "12345678910",
    "sku_id": [
        "BRO-70057-00002-00001",
        "BRO-70057-00002-00002",
    ],
    "caption": "bullshit",
    "video_id": "a66e342b-f4bf-471c-8810-e1157a8e677e",
    "video_path": "https://storage.googleapis.com/test-images-image-qc/Video_QC/sample_videos/(FMU)%20Buttonscarves%20Champ%20de%20Fleurs%20Voile%20Square%20-%20Tabebuya_Reels-REVISI.mp4",
}


# ----- FIXTURE: sample_video_qc_request -----
# A "fixture" is something pytest prepares for a test. If a test function has a parameter
# with the same name as a fixture (e.g. "sample_video_qc_request"), pytest will run this
# function and pass its return value into the test.
@pytest.fixture
def sample_video_qc_request():
    # We return a COPY of the sample so that if a test changes it (e.g. deletes a key),
    # other tests still get the original. Without .copy(), all tests would share one dict.
    return SAMPLE_VIDEO_QC_REQUEST.copy()


# ----- MOCK CLASSES (FAKE KAFKA) -----
# Our real app connects to Kafka (a message broker) when it starts. In tests we don't want
# to run a real Kafka server. So we create "mock" (fake) classes that have the same
# methods the app calls (start, stop, start_consumer) but do nothing real.


# Fake Kafka "consumer" (reads messages). The app only calls .start() and .stop() on it.
class MockConsumer:
    # async def = this function can "await" other async work (e.g. sleep). The app expects this.
    async def start(self):
        # pass = do nothing. We just need the method to exist so the app doesn't crash.
        pass

    async def stop(self):
        pass


# Fake Kafka "producer" (sends messages). Same idea: .start() and .stop() do nothing.
class MockProducer:
    async def start(self):
        pass

    async def stop(self):
        pass


# Fake "PubSub" (the class that wraps consumer + producer in our app).
# When the app does: pubsub = PubSub(...); await pubsub.consumer.start(); ...
# we want it to use OUR class instead of the real one, so no real Kafka is used.
class MockPubSub:
    # __init__ runs when we create MockPubSub(...). The app passes 3 arguments; we accept them
    # but don't use them for anything except storing (so the interface matches the real PubSub).
    def __init__(self, consumer_topics, producer_topic, process_batch_requests_from_kafka):
        # Instead of real Kafka consumer/producer, we use our fake ones.
        self.consumer = MockConsumer()
        self.producer = MockProducer()
        self.producer_topic = producer_topic
        self.process_batch_requests_from_kafka = process_batch_requests_from_kafka

    # The app starts a background task that runs start_consumer() forever. Our fake version
    # also runs "forever" (a loop that sleeps) until the app shuts down and cancels the task.
    async def start_consumer(self):
        try:
            # Loop forever, sleeping 1 second each time. This keeps the task "alive" like the real consumer loop.
            while True:
                await asyncio.sleep(1)
        # When the app shuts down, it cancels this task. Python raises CancelledError. We catch it.
        except asyncio.CancelledError:
            pass
        # "finally" runs whether we exit normally or by cancellation. The real app stops consumer/producer here too.
        finally:
            await self.consumer.stop()
            await self.producer.stop()


# ----- FIXTURE: mock_kafka (autouse) -----
# autouse=True means: run this fixture for EVERY test in this folder, even if the test
# doesn't list "mock_kafka" as a parameter. So every test automatically gets a fake Kafka.
@pytest.fixture(autouse=True)
def mock_kafka():
    # patch("where.to.patch", what.to.replace.with)
    # "application.app.pub_sub.PubSub" = the real PubSub class as seen from the app module.
    # When the app does "pub_sub.PubSub(...)", it will get MockPubSub instead of the real class.
    with patch("src.application.app.pub_sub.PubSub", MockPubSub):
        # yield = "pause here and run the test; when the test finishes, come back and remove the patch."
        # So: while any test is running, the patch is active. After the test, the real PubSub is restored.
        yield


# ----- FIXTURE: client -----
# This is the main fixture tests use to talk to our API. A test that has "client" as a parameter
# gets this fixture: a TestClient bound to our FastAPI app, so we can do client.get("/health") etc.
@pytest.fixture
def client():
    # We import the app here (inside the fixture) so that it is loaded AFTER mock_kafka has run.
    # That way when the app's startup code runs and creates PubSub(...), it gets MockPubSub.
    from src.application.app import app

    # TestClient(app) creates a fake HTTP client. "with ... as c" means: when we enter this block,
    # the app's "startup" event runs (e.g. connect to Kafka – but we mocked it!). When we exit
    # the block (after the test), the app's "shutdown" event runs (e.g. disconnect).
    with TestClient(app) as c:
        # yield c = give the test the client "c". The test runs. When the test ends, we exit
        # the "with" block and shutdown runs.
        yield c

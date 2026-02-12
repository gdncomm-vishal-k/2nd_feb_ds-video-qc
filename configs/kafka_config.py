consumer_config = {
    "bootstrap_servers": ["kafka-01.qa2-sg.cld:9092","kafka-02.qa2-sg.cld:9092"],
    "group_id": "ds_image_qc",
    "session_timeout_ms": 6000,
    "auto_offset_reset": "latest",
    "enable_auto_commit": False
}

producer_config = {
    "bootstrap_servers": ["kafka-01.qa2-sg.cld:9092","kafka-02.qa2-sg.cld:9092"]
}

# Understand the meaning of consumer_config and producer_config in detail

consumer_timeout = 100
consumer_topics = ["image_qc_request"]
producer_topic = "image_qc_response"

# Kafka will process in batches of 8 request at a time, need to test for best value.
batch_size = 8
# Kafka will wait for 6 seconds before rebalancing the consumer group.
rebalance_wait_seconds = 6
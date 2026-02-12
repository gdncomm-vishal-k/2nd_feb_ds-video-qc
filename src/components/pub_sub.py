import asyncio
import json
import logging as logger

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import IllegalStateError

from configs.kafka_config import consumer_config, producer_config, batch_size, rebalance_wait_seconds, consumer_timeout


class PubSub:
    """
    Handles Kafka consume → process → produce flow
    """

    def __init__(self, consumer_topics, producer_topic, process_batch_requests_from_kafka):
        logger.info("Initializing Kafka PubSub")

        # Kafka consumer
        self.consumer = AIOKafkaConsumer(
            *consumer_topics,
            **consumer_config
        )

        # Kafka producer
        self.producer = AIOKafkaProducer(
            **producer_config
        )

        self.producer_topic = producer_topic
        self.batch_size = batch_size

        # core logic to process the batch requests from kafka
        self.process_batch_requests_from_kafka = process_batch_requests_from_kafka

    async def process_messages(self, messages):
        """
        Process messages fetched from Kafka
        """
        if not messages:
            return

        for tp, msgs in messages.items():
            if not msgs:
                continue

            requests = []
            decode_results = []

            for msg in msgs:
                try:
                    data = json.loads(msg.value.decode("utf-8"))
                    requests.append(data)
                    decode_results.append(None)
                except json.JSONDecodeError:
                    decode_results.append({"error": "Invalid JSON format"})

            # Run batch processing for valid requests only
            try:
                model_results = await self.process_batch_requests_from_kafka(requests)
            except Exception:
                model_results = [
                    {"error": "Batch processing failed", "input_received": r}
                    for r in requests
                ]

            # One response per message: use decode error or corresponding model result
            responses = []
            j = 0
            for i in range(len(msgs)):
                if decode_results[i] is not None:
                    responses.append(decode_results[i])
                else:
                    responses.append(model_results[j])
                    j += 1

            await self.produce_and_commit(tp, msgs, responses)

    async def start_consumer(self):
        """
        Starts Kafka consumer loop
        """
        logger.info("Kafka consumer started")

        try:
            while True:
                try:
                    messages = await self.consumer.getmany(
                        timeout_ms=consumer_timeout,
                        max_records=self.batch_size,
                    )
                    await self.process_messages(messages)

                except IllegalStateError:
                    await asyncio.sleep(rebalance_wait_seconds)

        except asyncio.CancelledError:
            logger.info("Kafka consumer cancelled")

        finally:
            await self.consumer.stop()
            await self.producer.stop()
            logger.info("Kafka consumer stopped")
            exit()

    async def produce_and_commit(self, tp, msgs, result):
        """
        this method will produce the response and commits the offset.
        """
        logger.info("publishing {} responses to kafka".format(len(result)))
        for response in result:
            await self.producer.send(
                self.producer_topic, json.dumps(response).encode("utf-8")
            )
        logger.debug("kafka flushing")
        await self.producer.flush()
        logger.debug("committing offsets")
        await self.consumer.commit({tp: msgs[-1].offset + 1})
        logger.info("offsets committed")

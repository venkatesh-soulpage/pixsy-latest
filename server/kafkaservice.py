import json

from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import uuid
from ml import predict_data

# BOOTSTRAP_SERVERS = "kafka:9093"
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "vectors"


""" Kafka endpoints """


def kafkaconsumer(photo_id, matches_id):
    consumer = KafkaConsumer(bootstrap_servers=BOOTSTRAP_SERVERS)

    tp = TopicPartition(TOPIC_NAME, 0)
    # register to the topic
    consumer.assign([tp])

    # obtain the last offset value
    consumer.seek_to_end(tp)
    lastOffset = consumer.position(tp)
    consumer.seek_to_beginning(tp)

    photo_data = None
    matches_data = None
    for message in consumer:
        consumer_message = message
        message = json.loads(message.value)
        if message.get("request_id") == photo_id:
            photo_data = message.get("data")
        if message.get("request_id") == matches_id:
            matches_data = message.get("data")
        if consumer_message.offset == lastOffset - 1:
            break
    consumer.close()

    return predict_data(photo_data, matches_data)


def kafkaproducer(data, im_type, message_id):
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
    message = {
        "request_id": message_id,
        "data": data,
        "type": im_type,
    }

    producer.send(TOPIC_NAME, json.dumps(message).encode("utf-8"))
    producer.flush()
    producer.close()
    return None

import logging, json, sys
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": self.formatTime(record,"%Y-%m-%dT%H:%M:%S"), "level":record.levelname, "name":record.name, "message":record.getMessage()}
        for k in ("run_id","doc_id","chunk_id"):
            v = getattr(record, k, None)
            if v: data[k] = v
        return json.dumps(data)
def setup_logging(level: int=logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])

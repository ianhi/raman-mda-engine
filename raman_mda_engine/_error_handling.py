import traceback

import requests
import wrapt

_url = ""


def set_webhook_url(url):
    global _url
    _url = url


@wrapt.decorator
def slack_notify(wrapped, instance, args, kwargs):
    try:
        return wrapped(*args, **kwargs)
    except Exception as e:
        if _url != "":
            tb = "".join(traceback.TracebackException.from_exception(e).format())
            data = {"text": f"Something broke! <!channel>\n```\n{tb}\n```"}
            requests.post(_url, json=data)
        raise e

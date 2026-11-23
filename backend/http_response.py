"""Send cookie headers intact through the Azure Functions Python HTTP binding."""

import azure.functions as func


class CookieResponse(func.HttpResponse):
    """Keep lower-case raw headers instead of using the binding's cookie converter.

    azure.functions.http.HttpResponseConverter otherwise parses Set-Cookie with
    Python 3.11's SimpleCookie (which rejects Partitioned), then serializes a
    restricted cookie object. HTTP header names are case-insensitive on the wire;
    a lower-case dict preserves the raw header through the binding's headers path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_headers = {key.lower(): value for key, value in super().headers.items()}

    @property
    def headers(self) -> dict[str, str]:
        return self._raw_headers

import inspect
import json
import re
from inspect import Parameter
from json import JSONDecodeError
from typing import Any, Callable, Awaitable, Optional, Mapping
from urllib.parse import parse_qs, urlsplit

import requests
from huggingface_hub import scale_to_zero_inference_endpoint
from paddle.base.libpaddle.eager.ops.legacy import data_norm
from urllib3 import request

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class Connection:
    def __init__(self, scope: Scope, receive: Receive, send: Send):
        self.scope: Scope = scope
        self.receive: Receive = receive
        self.send: Send = send

    async def send_response(self, response: "HttpResponse"):
        await self.send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": response.headers,
            }
        )
        await self.send(
            {
                "type": "http.response.body",
                "body": response.body.encode(),
                "more_body": False,
            }
        )


class HttpResponse:
    def __init__(
        self,
        body: Optional[bytes] | Optional[str] = "",
        status_code: int = 200,
        headers: Optional[Mapping[str, str]] = None,
    ):
        self.body = body
        self.status_code = status_code
        self.headers = (
            headers
            if headers
            else [
                [b"content-type", b"text/plain"],
            ]
        )

    def __repr__(self):
        return f"HttpResponse(status_code={self.status_code}, body={self.body})"


class JSONResponse(HttpResponse):
    def __init__(self, body: Any, status_code: int = 200):
        super().__init__(body=json.dumps(body), status_code=status_code)

    def __repr__(self):
        return f"JSONResponse(status_code={self.status_code}, body={self.body})"


class Endpoint:
    def __init__(
        self,
        path: str,
        method: str,
        func: Callable,
        path_args: Mapping[str, type],
        query_args,
    ):
        self.path = path
        self.method = method
        self.func = func
        self.path_args = path_args
        self.query_args = query_args

    def __repr__(self):
        return (
            f"Endpoint(path={self.path}, method={self.method}, path_args={self.path_args}, "
            f"query_args={self.query_args}, func={self.func}"
        )


class Registerer:
    def __init__(self):
        self._endpoints: list[Endpoint] = []

    def add_endpoint(self, path: str, method: str, func: Callable):
        pattern = re.compile(r"{(\w+)}")
        args = re.findall(pattern, path)
        signature = inspect.signature(func).parameters
        query_args = []
        path_args = []
        for arg in signature:
            if arg in args:
                path_args.append(signature[arg])
            else:
                query_args.append(signature[arg])
        ep = Endpoint(
            path.split("/")[1],
            method,
            func,
            {p.name: p.annotation for p in path_args},
            query_args,
        )
        self._endpoints.append(ep)

    def register(self, path: str, method: str):
        def decorator(func: Callable):
            self.add_endpoint(path, method, func)
            return func

        return decorator

    @property
    def endpoints(self) -> list[Endpoint]:
        return self._endpoints


class MyApp:
    def __init__(self, registerer: Registerer):
        self._registerer = registerer
        self.connection: Optional[Connection] = None

    def __call__(self, *args, **kwargs):
        async def _handle(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                self.connection = Connection(scope, receive, send)
                await self._handle_http()
            else:
                raise NotImplementedError("Only http is supported")

        return _handle

    async def _handle_http(self) -> None:
        body = await self.connection.receive()
        try:
            data = json.loads(body.get("body", b"[]"))
        except JSONDecodeError:
            data = None
        try:
            endpoint, func_args = self.find_endpoint(
                self.connection.scope["path"],
                self.connection.scope["query_string"],
                self.connection.scope["method"],
                data,
            )
            answer = await endpoint.func(**func_args)
            await self.connection.send_response(answer)
            return
        except (ValueError, TypeError, JSONDecodeError) as e:
            await self.connection.send_response(HttpResponse(status_code=422))
            return
        except NotImplementedError:
            await self.connection.send_response(HttpResponse(status_code=404))
            return

    def find_endpoint(
        self, path: str, query_string: str, method: str, data: dict
    ) -> (Endpoint, dict[str, Any]):
        urlst = [x for x in urlsplit(path).path.split("/") if x != ""]
        ep_path = urlst[0] if len(urlst) > 0 else ""
        path_args = urlst[1:] if len(urlst) > 1 else []
        query_args = parse_qs(query_string)
        query_args = {k.decode(): v[0] for k, v in query_args.items()}

        for endpoint in self._registerer.endpoints:
            if endpoint.path == ep_path and endpoint.method == method:
                func_args = {}
                for i, arg in enumerate(path_args):
                    arg_name = list(endpoint.path_args.keys())[i]
                    func_args[arg_name] = endpoint.path_args[arg_name](arg)
                for i, (name, value) in enumerate(query_args.items()):
                    if name in query_args:
                        func_args[name] = endpoint.query_args[i].annotation(value)
                    # ?else:
                    # ?    func_args[name] = endpoint.query_args[i].default
                if data is not None:
                    diff = [x for x in endpoint.query_args if x not in func_args.keys()]
                    for p in diff:
                        func_args[p.name] = data
                return endpoint, func_args
        raise NotImplementedError("Not found")


if __name__ == "__main__":
    HOST = "localhost"
    PORT = 8000
    BASE_URL = f"http://{HOST}:{PORT}"
    ans = requests.get(BASE_URL + "/mean", json=[1, 2, 3])
    print(ans.content)

from functools import reduce

from src.MyAPI.MyAPI import MyApp, Registerer, HttpResponse, JSONResponse

regesterer = Registerer()
app = MyApp(regesterer)


@regesterer.register("/factorial/", "GET")
async def factorial(n: int) -> HttpResponse:
    if n < 0:
        return HttpResponse(status_code=400)
    result = reduce(lambda x, y: x * y, range(1, n + 1), 1)
    return JSONResponse({"result": str(result)})


@regesterer.register("/fibonacci/{n}", "GET")
async def fibonacci(n: int) -> HttpResponse:
    if n < 0:
        return HttpResponse(status_code=400)
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return JSONResponse({"result": str(a)})


@regesterer.register("/mean/", "GET")
async def mean(array: list[float | int]) -> HttpResponse:
    if len(array) == 0:
        return HttpResponse(status_code=400)
    result = sum(array) / len(array)
    return JSONResponse({"result": str(result)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost")

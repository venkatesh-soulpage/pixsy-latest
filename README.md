To start the server using uvicorn
try: `uvicorn server.app:app`

Upload single photo and multiple match photos to get unique id of photo and matches, which is an request id of kafka consumer id

Step-1:
URL: `http://localhost:8000/`
Method: `POST`
request-headers: `multipart`

payload:

```
photo[]: "path/to/file/",
macthes[]: "path/to/file/",
macthes[]: "path/to/file/"
```

Response:

```
{
  "photo_id": "189f0a5c-1fa4-4d99-a7d1-2aefe602564a",
  "matches_id": "9974b1db-7f78-41b9-a82e-c60a4ba8cf11"
}
```

Using kafka consumer id's, we can get the image vectors to predict their scores.

Step-2:

URL: `http://localhost:8000/predict`
Method: `POST`

payload:

```
{
  "photo_id": "189f0a5c-1fa4-4d99-a7d1-2aefe602564a",
  "matches_id": "9974b1db-7f78-41b9-a82e-c60a4ba8cf11"
}
```

Response:

```
[
  {
    "photo_id": "5a9f7bd9b8254d0040c2ee95.jpg",
    "match_id": "5a9f7bd9b8254d0040c2ee95_exact.jpg",
    "FP_Prediction": 12,
    "Score": 0.9999355674
  },
  {
    "photo_id": "5a9f7bd9b8254d0040c2ee95.jpg",
    "match_id": "5a9808e2ae69cf0ae72aa6fd.jpg",
    "FP_Prediction": 10,
    "Score": 0.295502007
  }
]
```

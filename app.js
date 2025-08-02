import express from "express"

const app = express()

app.set("view engine", "ejs")

app.use(logger)
app.use(express.urlencoded({ extended: true }))
app.use(express.json())


app.get("/", (req, res) => {
  res.render("index", {})
})

function logger(req, res, next) {
  console.log("There is a request to", req.originalUrl)
  next()
}

const port = 3000
app.listen(port, () => {
  console.log(`App is running on http://localhost:${port} | http://172.16.30.145:${port}/`)
})

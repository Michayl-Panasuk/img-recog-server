import * as tf from "@tensorflow/tfjs-node";
import coco_ssd from "@tensorflow-models/coco-ssd";

// * Server Stuff
import express from "express";
import busboy from "busboy";
import { config } from "dotenv";
config()


let model = undefined;
(async () => {
  model = await coco_ssd.load({
    base: "mobilenet_v1"
  })
})();


export const app = express();

const PORT = process.env.PORT || 5000
app.post("/predict", (req, res) => predict(req, res))

app.listen(PORT, () => {
  console.log("Listening on port " + PORT)
})

function predict(req, res) {
  if (!model) {
    res.status(500).send("Model is not loaded yet!");
  }
  const bb = busboy({ headers: req.headers });
  bb.on("file", (fieldname, file, filename, encoding, mimetype) => {
    const buffer = [];
    file.on("data", (data) => {
      buffer.push(data);
    });
    file.on("end", async () => {
      // * Run Object Detection
      const image = tf.node.decodeImage(Buffer.concat(buffer));
      const predictions = await model.detect(image, 3, 0.25);
      res.json(predictions);
    });
  })
  req.pipe(bb)
};
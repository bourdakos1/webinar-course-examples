import React, { useCallback, useRef, useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import * as tf from '@tensorflow/tfjs'

import { useCanvasBrush } from './canvas-utils'
import './index.css'

// `App` gets re-run any time a state variable changes, in this case `model` and
// `prediction`.
const App = () => {
  // State variables.
  const [model, setModel] = useState(undefined)
  const [prediction, setPrediction] = useState(0)

  // Create a reference to the canvas element and pass it to `useCanvasBrush`.
  // This turns our canvas into a drawing pad.
  const canvasRef = useRef(null)
  const canvasBrush = useCanvasBrush(canvasRef)

  // `useEffect` is part of React, the code inside of this effect only gets run
  // once when the webpage is first loaded.
  useEffect(() => {
    // Load our model into the component's state.
    const loadModel = async () => {
      const model = await tf.loadLayersModel(
        `${process.env.PUBLIC_URL}/mnist_model/model.json`
      )
      setModel(model)
    }
    loadModel()
  }, [])

  // When the `clear` button is pressed, clear the canvas.
  const handleClear = useCallback(() => {
    canvasBrush.clear()
  }, [canvasBrush])

  //////////////////////////////////////////////////////////////////////////////
  // Where the magic happens
  //////////////////////////////////////////////////////////////////////////////
  // When the `predict` button is pressed, make a prediction with TensorFlow.js.
  const handlePredict = useCallback(() => {
    // Anything inside `tidy` gets disposed (prevents memory leaks).
    const batched = tf.tidy(() => {
      // Turn the image on the canvas into a `tensor` (pretty much a fancy name
      // for an array or an array of arrays). Our data needs to be in the form
      // of a `tensor` to be able to pass it through our model.
      const img = tf.browser.fromPixels(canvasRef.current)

      // When our image is stored as a tensor, it's essentially a giant array
      // where each item in the array is a pixel's value. Each pixel has a value
      // from 0 to 255:
      // - 0 meaning no light is emitted from the pixel (black/darkness).
      // - 255 meaning the pixel is at full brightness (white/bright).
      //
      // Our model is actually expecting `1` for places where we have drawn and
      // `0` for where we haven't. We have a bit of an issue since our pen color
      // is black (or 0) and our background is white (or 255). We can divide the
      // values by 255 so our values go from 0 to 1. However, our values are
      // still backwards, because we need white to be 0 and black to be 1. We
      // can fix this by subtracting by 1 and dividing by -1.
      // Or simplified:
      // 255 - 255 =    0   =>     0 / -255 = 0.0
      // 127 - 255 = -128   =>  -128 / -255 = 0.5
      //   0 - 255 = -255   =>  -255 / -255 = 1.0
      const inverted = img.sub(tf.scalar(255)).div(tf.scalar(-255))

      // Scale our 280 pixel (px) image to 28px.
      const scaledImg = tf.image.resizeBilinear(inverted, [28, 28], true)

      // Right now our image is in a tensor of shape [28, 28, 3]. What is that
      // `3` in the third position of the tensor? I fibbed a little when I said
      // each pixel is a value from 0 to 255. Each pixel is actually 3 values
      // from 0 to 255, one for each color (red, green, blue). The 3 colors are
      // blended together to make all the colors of the rainbow.
      // (https://y6n4s6w9.stackpathcdn.com/wp-content/uploads/2018/02/rgb_model.gif)
      // In reality black is [0, 0, 0] and white is [255, 255, 255].
      // A simple way to flatten this array into a single value is to take the
      // mean of the image tensor across the 3rd axis. This will turn the image
      // into a single channel ([28, 28]) greyscale image.
      // For example:
      // (255 + 255 + 255) / 3 = 255
      // [255, 255, 255] => 255
      //
      // Why don't we just slice the array and only take one of the values?
      // - This would work perfectly in our case because the image is already
      //   black and white. However, say we just look at the first value, if we
      //   changed our pen color to red ([255, 0, 0]) we would get 255 (same
      //   color as white).
      //
      // Side note about actual greyscale conversion:
      // Most people don't take the mean of the image because it weights each
      // color equally with 33%. When in reality certain colors, like green,
      // don't look as dark as say, blue. A common weighting is:
      // 0.2126 * red + 0.7152 * green + 0.0722 * blue
      // (https://en.wikipedia.org/wiki/Grayscale)
      const greyscale = scaledImg.mean(2) // 3rd axis (2, because indexed by 0)

      // Resize the input from [28, 28] to [1, 28, 28]. The input of our model
      // has a shape of [?, 28, 28] because it supports multiple images at once.
      // We are essentially adding our single image to an empty array.
      return greyscale.expandDims(0)
    })

    const best = tf.tidy(() => {
      const predictions = model.predict(batched).dataSync()
      return tf
        .tensor1d(predictions)
        .argMax()
        .dataSync()
    })

    setPrediction(best)
  }, [model])
  //////////////////////////////////////////////////////////////////////////////

  // The html that gets rendered. Our canvas, 2 buttons and the prediction.
  return (
    <div>
      <canvas
        style={{ border: '1px solid black' }}
        ref={canvasRef}
        width="280"
        height="280"
      />
      <div>{prediction}</div>
      <button onClick={handleClear}>clear</button>
      <button onClick={handlePredict}>predict</button>
    </div>
  )
}

ReactDOM.render(<App />, document.getElementById('root'))

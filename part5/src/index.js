import React, { useCallback, useRef, useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import * as tf from '@tensorflow/tfjs'

import { useCanvasBrush } from './canvas-utils'
import './index.css'

const App = () => {
  const [model, setModel] = useState(undefined)
  const [prediction, setPrediction] = useState(0)
  const canvasRef = useRef(null)
  const canvasBrush = useCanvasBrush(canvasRef)

  useEffect(() => {
    const loadModel = async () => {
      const model = await tf.loadLayersModel(
        `${process.env.PUBLIC_URL}/mnist_model/model.json`
      )
      setModel(model)
    }
    loadModel()
  }, [])

  const handleClear = useCallback(() => {
    canvasBrush.clear()
  }, [canvasBrush])

  const handlePredict = useCallback(() => {
    // Anything inside tidy gets disposed of, prevents memory leaks
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(canvasRef.current)
      const inverted = img.sub(tf.scalar(255)).div(tf.scalar(-255))
      const scaledImg = tf.image.resizeBilinear(inverted, [28, 28], true)
      const greyscale = scaledImg.mean(2)

      // Reshape to a single-element batch so we can pass it to executeAsync.
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

import { useCallback, useEffect, useState } from 'react'

const draw = (canvas, coordinates) => {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  // Fill with white, otherwise the tensor will be all zeros.
  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  ctx.strokeStyle = 'black'
  ctx.lineJoin = 'round'
  ctx.lineWidth = 30

  coordinates.forEach((_, i) => {
    ctx.beginPath()
    if (coordinates[i][2] && i) {
      ctx.moveTo(coordinates[i - 1][0], coordinates[i - 1][1])
    } else {
      ctx.moveTo(coordinates[i][0] - 1, coordinates[i][1])
    }
    ctx.lineTo(coordinates[i][0], coordinates[i][1])
    ctx.closePath()
    ctx.stroke()
  })
}

export const useCanvasBrush = canvasRef => {
  const [coordinates, setCoordinates] = useState([])
  const [isDrawing, setIsDrawing] = useState(false)

  const calculateCoordinates = useCallback(
    (e, drawing) => {
      const x = e.pageX - canvasRef.current.offsetLeft
      const y = e.pageY - canvasRef.current.offsetTop
      setCoordinates(coordinates => [...coordinates, [x, y, drawing]])
    },
    [canvasRef]
  )

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false)
  }, [])

  const handleMouseDown = useCallback(
    e => {
      setIsDrawing(true)
      calculateCoordinates(e)
    },
    [calculateCoordinates]
  )

  const handleMouseMove = useCallback(
    e => {
      if (isDrawing) {
        calculateCoordinates(e, true)
      }
    },
    [calculateCoordinates, isDrawing]
  )

  useEffect(() => {
    draw(canvasRef.current, coordinates)
  }, [canvasRef, coordinates])

  useEffect(() => {
    document.addEventListener('mouseup', handleMouseUp)
    let currentCanvas
    if (canvasRef.current) {
      currentCanvas = canvasRef.current
      canvasRef.current.addEventListener('mousedown', handleMouseDown)
      canvasRef.current.addEventListener('mousemove', handleMouseMove)
    }
    return () => {
      document.removeEventListener('mouseup', handleMouseUp)
      currentCanvas.removeEventListener('mousedown', handleMouseDown)
      currentCanvas.removeEventListener('mousemove', handleMouseMove)
    }
  }, [canvasRef, handleMouseDown, handleMouseMove, handleMouseUp])

  return {
    clear: () => {
      setCoordinates([])
      const context = canvasRef.current.getContext('2d')
      context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    }
  }
}

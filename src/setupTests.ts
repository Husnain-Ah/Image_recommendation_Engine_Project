import '@testing-library/jest-dom';

jest.mock('@tensorflow/tfjs', () => ({
  tensor3d: jest.fn((values, shape) => ({
    shape,
    values,
    dispose: jest.fn()
  })),
  tensor1d: jest.fn((data) => ({
    data,
    dispose: jest.fn(),
    arraySync: jest.fn(() => data),
    dataSync: jest.fn(() => data)
  })),
  scalar: jest.fn((value) => ({
    value,
    mul: jest.fn((other) => other),
    dataSync: jest.fn(() => [value])
  })),
  mul: jest.fn((a, b) => ({
    data: [1, 2, 3],
    dispose: jest.fn(),
    dataSync: jest.fn(() => [0.5])
  })),
  sum: jest.fn(() => ({
    data: [0.5],
    div: jest.fn(() => ({ 
      arraySync: () => 0.5,
      dataSync: () => [0.5]
    }))
  })),
  norm: jest.fn(() => ({
    mul: jest.fn(() => ({ 
      data: [1],
      dataSync: () => [1]
    }))
  })),
  browser: {
    fromPixels: jest.fn()
  }
}));

jest.mock('@tensorflow-models/mobilenet', () => ({
  load: jest.fn().mockResolvedValue({
    classify: jest.fn().mockResolvedValue([
      { className: 'dog', probability: 0.8 },
      { className: 'cat', probability: 0.2 }
    ]),
    infer: jest.fn().mockReturnValue({
      data: [1, 2, 3],
      dispose: jest.fn()
    })
  })
}));
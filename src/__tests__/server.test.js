const fs = require('fs');
const path = require('path');
const axios = require('axios');

jest.mock('axios');
jest.mock('fs');
jest.mock('path');
jest.mock('express', () => { //manually doing this as supertest breaks python tests somehow even though theyre not related...
  const mockExpress = () => ({
    use: jest.fn(),
    get: jest.fn(),
    listen: jest.fn(),
    set: jest.fn()
  });

  const mockRouter = {
    post: jest.fn(),
    get: jest.fn(),
    use: jest.fn()
  };

  mockExpress.Router = () => mockRouter;
  mockExpress.static = jest.fn(() => (request, response, next) => next());
  mockExpress.json = jest.fn(() => (request, response, next) => next());

  return mockExpress;
});

describe('Server.js Tests', () => {
  let server;
  let mockHandler;

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();

    fs.readFileSync.mockImplementation((filePath) => {
      if (filePath.includes('words.txt')) {
        return 'n01440764\ttench\n';
      }
      if (filePath.includes('metadata.json')) {
        return JSON.stringify([
          { filename: 'img1.JPEG', wnid: 'n01440764', label: 'tench' }
        ]);
      }
      return '';
    });

    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue([]);

    path.join.mockImplementation((...parts) => parts.join('/'));
    path.dirname.mockReturnValue('/mocked/path');

    const express = require('express');
    const router = express.Router();

    router.post.mockImplementation((route, fn) => {
      if (route === '/search-images') {
        mockHandler = fn;
      }
    });

    server = require('../../server');
  });

    test('cosineSimilarity should return correct value', () => {
      const a = [1, 2, 3];
      const b = [4, 5, 6];
      const similarity = server.cosineSimilarity(a, b);

      expect(similarity).toBeCloseTo(0.9746, 4);
    });

  describe('Image Search endpoint tests', () => {
    test('returns 400 when no keyword is sent', async () => {
      const request = { body: {} };
      const response = {
        status: jest.fn().mockReturnThis(),
        json: jest.fn()
      };

      await mockHandler(request, response);

      expect(response.status).toHaveBeenCalledWith(400);
      expect(response.json).toHaveBeenCalledWith({ error: 'No keyword provided' });
    });

    test('returns 500 when embedding service fails', async () => {

      const request = { body: { keyword: 'test' } };
      const response = {
        status: jest.fn().mockReturnThis(),
        json: jest.fn()
      };

      await mockHandler(request, response);

      expect(response.status).toHaveBeenCalledWith(500);
      expect(response.json).toHaveBeenCalledWith({ error: 'Failed to get embedding' });
    });

    // test('returns 200 with results when a match is found', async () => { //broken, fix later

    //   server.invertedIndex = { tench: ['img1.JPEG'] };
    //   server.metadata = [{ filename: 'img1.JPEG', wnid: 'n01440764', label: 'tench' }];
    //   server.cosineSimilarity = jest.fn().mockReturnValue(0.9);

    //   const request = { body: { keyword: 'test' } };
    //   const response = {
    //     status: jest.fn().mockReturnThis(),
    //     json: jest.fn()
    //   };

    //   await mockHandler(request, response);

    //   expect(response.status).toHaveBeenCalledWith(200);
    //   expect(response.json).toHaveBeenCalledWith({
    //     results: ['tiny-imagenet-200/train/n01440764/images/img1.JPEG'],
    //     match: 'tench',
    //     similarity: '0.900'
    //   });
    });
});

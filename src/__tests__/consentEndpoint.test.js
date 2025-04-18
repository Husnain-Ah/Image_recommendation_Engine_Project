const fs = require('fs');
const path = require('path');
const axios = require('axios');

jest.mock('axios');
jest.mock('fs', () => ({
  existsSync: jest.fn(),
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
}));
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

describe('Server.js consent endpoint Tests', () => {
  let server;
  let mockHandler;

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();

    const express = require('express');
    const router = express.Router();

    router.post.mockImplementation((route, fn) => {
      if (route === '/consent') {
        mockHandler = fn; 
      }
    });

    server = require('../../server');
  });

 
    test('returns 200 and stores ratings successfully', async () => {
     
      const mockRatings = [{ relevant: 0, user_rating: 2, timestamp: 'later' }];
      const request = { body: { ratings: mockRatings } };
      const response = {
        status: jest.fn().mockReturnThis(),
        json: jest.fn()
      };

      await mockHandler(request, response);

      expect(response.status).toHaveBeenCalledWith(200);
      expect(response.json).toHaveBeenCalledWith({ message: 'Ratings stored successfully' });
    });

    test('returns 400 if invalid JSON exists in the ratings file', async () => {
    
      const request = { body: {  } };
      const response = {
        status: jest.fn().mockReturnThis(),
        json: jest.fn(),
      };

      await mockHandler(request, response);

      expect(response.status).toHaveBeenCalledWith(400);
      expect(response.json).toHaveBeenCalledWith({ error: 'Invalid ratings data' });
    });

});
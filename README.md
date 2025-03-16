# Emotion Detection Game - Sant Nirankari Mission

A real-time emotion detection game that creates ripple effects based on facial expressions and provides spiritual teachings.

## Prerequisites

- Python 3.8+
- Webcam
- Required model files:
  - `src/model.h5`
  - `src/haarcascade_frontalface_default.xml`
  - `src/game-music.mp3` (optional)

## Local Development

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
cd src
python web_app.py
```

## Production Deployment

### Option 1: Deploy to Heroku

1. Install Heroku CLI and login:

```bash
heroku login
```

2. Create new Heroku app:

```bash
heroku create your-app-name
```

3. Configure buildpacks:

```bash
heroku buildpacks:add --index 1 heroku/python
```

4. Deploy:

```bash
git push heroku main
```

### Option 2: Deploy using Docker

1. Build the Docker image:

```bash
docker build -t emotion-detection-game .
```

2. Run the container:

```bash
docker run -p 8000:8000 emotion-detection-game
```

### Option 3: Deploy to a VPS/Cloud Server

1. SSH into your server
2. Clone the repository
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run with gunicorn:

```bash
gunicorn --chdir src web_app:app -b 0.0.0.0:8000
```

## Production Testing

1. **Load Testing:**

   - Use Apache Benchmark or wrk for load testing:

   ```bash
   ab -n 1000 -c 50 https://your-domain.com/
   ```

2. **Browser Testing:**

   - Test on different browsers (Chrome, Firefox, Safari)
   - Test on mobile devices
   - Verify webcam permissions work correctly

3. **Error Handling:**

   - Test with webcam disconnected
   - Test with slow internet connection
   - Verify error messages are displayed properly

4. **Performance Monitoring:**
   - Set up monitoring using New Relic or Datadog
   - Monitor CPU and memory usage
   - Track response times and error rates

## Security Considerations

1. Enable HTTPS using SSL/TLS certificates
2. Implement rate limiting
3. Add CORS headers if needed
4. Ensure proper webcam permission handling
5. Monitor for suspicious activities

## Environment Variables

Configure these environment variables in production:

- `FLASK_ENV=production`
- `FLASK_DEBUG=0`
- `ALLOWED_ORIGINS=your-domain.com`

## Maintenance

1. Regularly update dependencies
2. Monitor error logs
3. Backup model files
4. Update SSL certificates when needed

## Support

For issues and support, please create an issue in the repository or contact the maintainers.

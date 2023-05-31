from skylearn import app
import image_processing
if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5002)
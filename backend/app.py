from flask import Flask, request, jsonify
import tiktok
import threads

# Create Flask application instance
app = Flask(__name__)

@app.route('/request', methods=['POST'])
def handle_request():
    """Endpoint to verify and process URLs from TikTok and Threads"""
    try:
        # Get URL from the request
        data = request.get_json()
        
        # Check if data exists and contains 'url' key
        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing URL in request'
            }), 400
        
        url = data['url']
        
        # Check if URL is empty or contains only whitespace
        if not url or not url.strip():
            return jsonify({
                'success': False,
                'error': 'Empty URL'
            }), 400
        
        # Check if it's a TikTok URL
        if tiktok.is_tiktok_url(url):
            result = tiktok.download_tiktok_video(url, preferred_browser='firefox', quality='worst', custom_filename="temp_video.mp4")
            return result  # Return the result from TikTok processing
        
        # Check if it's a Threads URL
        if threads.is_threads_url(url):
            result = threads.parse_threads(url)
            return result  # Return the result from Threads processing
        
        # URL doesn't belong to any supported platform
        return jsonify({
            'success': False,
            'error': 'Unsupported URL. We only support TikTok and Threads.'
        }), 400
        
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            'success': False,
            'error': f'Error during verification: {str(e)}'
        }), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
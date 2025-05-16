# test_moviepy.py
try:
    from moviepy.editor import VideoFileClip
    print("moviepy imported successfully!")
    # Test FFmpeg integration
    video = VideoFileClip("path/to/a/sample/video.mp4")  # Replace with a sample video path
    print("Video loaded successfully!")
    video.close()
except Exception as e:
    print(f"Error: {e}")
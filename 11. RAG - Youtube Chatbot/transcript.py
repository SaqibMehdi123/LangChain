from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "J5_-l7WIO_w"

api = YouTubeTranscriptApi()

# Fetch transcript (will get Hindi since English not available for this video)
try:
    transcript = api.fetch(video_id, languages=['en'])
except:
    print("English not available, fetching Hindi transcript...")
    transcript = api.fetch(video_id, languages=['hi'])

text = " ".join([t.text for t in transcript])
print(text[:500])

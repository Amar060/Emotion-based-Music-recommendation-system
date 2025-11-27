import random

def recommend_music(emotion):
    mood_music = {
        'happy': ['Happy – Pharrell Williams', 'Can’t Stop The Feeling – Justin Timberlake', 'On Top of the World – Imagine Dragons'],
        'sad': ['Someone Like You – Adele', 'Let Her Go – Passenger', 'Fix You – Coldplay'],
        'angry': ['In The End – Linkin Park', 'Stronger – Kanye West', 'Smells Like Teen Spirit – Nirvana'],
        'fear': ['Demons – Imagine Dragons', 'Boulevard of Broken Dreams – Green Day'],
        'surprise': ['Adventure of a Lifetime – Coldplay', 'Thunder – Imagine Dragons'],
        'disgust': ['Believer – Imagine Dragons', 'Radioactive – Imagine Dragons'],
        'neutral': ['Perfect – Ed Sheeran', 'Stay – Justin Bieber']
    }
    return random.choice(mood_music.get(emotion, ['Shape of You – Ed Sheeran']))
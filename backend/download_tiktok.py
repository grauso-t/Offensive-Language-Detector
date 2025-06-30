import yt_dlp
import os

def scarica_video_tiktok(link):
    # Crea la cartella se non esiste
    directory = "Tiktok Downloads"
    if not os.path.exists(directory):
        os.makedirs(directory)

    opzioni = {
        'outtmpl': os.path.join(directory, '%(title)s.%(ext)s'),  # salva nella cartella specifica
        'format': 'mp4',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(opzioni) as ydl:
        try:
            print(f"🔗 Scaricamento in corso: {link}")
            ydl.download([link])
            print(f"✅ Download completato. File salvato in: {directory}")
        except Exception as e:
            print(f"❌ Errore durante il download: {e}")

if __name__ == "__main__":
    # Inserire link tiktok
    link_video = "https://www.tiktok.com/@chhengoutdoors168/video/7515312906431499540?is_from_webapp=1&sender_device=pc"
    scarica_video_tiktok(link_video)
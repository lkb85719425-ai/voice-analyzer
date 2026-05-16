import os
import uuid
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import librosa
from scipy.spatial.distance import cdist
import requests

# ---------- API 配置 ----------
LASTFM_API_KEY = "0c59f0b08fda6f813abf4abf00a19b6e"          # 你的 Last.fm 密钥
SPOTIFY_CLIENT_ID = "你的Spotify_Client_ID"                   # 可选
SPOTIFY_CLIENT_SECRET = "你的Spotify_Client_Secret"           # 可选

# 网易云公开 API（多镜像自动切换）
NETEASE_MIRRORS = [
    "https://music-api.heheda.top",
    "https://api.cenguigui.cn",
    "https://neteasecloudmusicapi.vercel.app"
]

# 离线相似歌手映射（API 全部失败时使用）
OFFLINE_SIMILAR = {
    "周杰伦": ["林俊杰", "王力宏", "陶喆", "陈奕迅"],
    "邓丽君": ["王菲", "蔡琴", "梅艳芳", "陈慧娴"],
    "Taylor Swift": ["Adele", "Ed Sheeran", "Katy Perry", "Lady Gaga"],
    "Adele": ["Taylor Swift", "Sam Smith", "Beyoncé", "Sia"],
    "陈奕迅": ["周杰伦", "林俊杰", "张学友", "李荣浩"],
    "王菲": ["邓丽君", "莫文蔚", "林忆莲", "孙燕姿"],
    "Bruno Mars": ["Justin Timberlake", "Pharrell Williams", "The Weeknd", "Charlie Puth"]
}

# ---------- 可选 Spotify ----------
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    if SPOTIFY_CLIENT_ID != "你的Spotify_Client_ID" and SPOTIFY_CLIENT_SECRET != "你的Spotify_Client_Secret":
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
    else:
        sp = None
except Exception:
    sp = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- 本地歌手特征库（26维）----------
def load_star_features():
    if os.path.exists('star_features.json'):
        with open('star_features.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "周杰伦": [-200, 30, -50, 20, -10, 5, -2, 8, -15, 10, -5, 3, 0, 0.1, 0.3, 0.2, 0.1, 0.05, 0.01, 0.02],
        "邓丽君": [-180, 50, -70, 40, -20, 15, -10, 12, -5, 20, -15, 8, 2, 0.3, 0.5, 0.1, 0.2, 0.15, 0.02, 0.05],
        "Taylor Swift": [-150, 20, -30, 10, -5, 2, 0, 5, -8, 8, -3, 1, 1, 0.25, 0.4, 0.15, 0.18, 0.08, 0.03, 0.04],
        "Adele": [-220, 10, -20, 5, 0, -2, 2, 3, -10, 5, 0, 0, 0, 0.2, 0.25, 0.3, 0.05, 0.1, 0.02, 0.03],
        "陈奕迅": [-190, 40, -60, 30, -15, 10, -8, 10, -12, 15, -10, 5, 1, 0.15, 0.35, 0.22, 0.12, 0.06, 0.04, 0.01],
        "王菲": [-170, 60, -80, 45, -25, 18, -12, 15, -10, 22, -18, 10, 3, 0.35, 0.55, 0.18, 0.25, 0.18, 0.03, 0.06],
        "Bruno Mars": [-160, 25, -35, 15, -8, 4, 0, 6, -7, 9, -4, 2, 1, 0.28, 0.42, 0.16, 0.19, 0.07, 0.03, 0.04]
    }

STAR_DB = load_star_features()

# ---------- 本地歌曲库 ----------
SONG_DB = [
    {"title": "晴天", "artist": "周杰伦", "low_midi": 57, "high_midi": 69},
    {"title": "月亮代表我的心", "artist": "邓丽君", "low_midi": 55, "high_midi": 72},
    {"title": "Love Story", "artist": "Taylor Swift", "low_midi": 59, "high_midi": 74},
    {"title": "Someone Like You", "artist": "Adele", "low_midi": 53, "high_midi": 70},
    {"title": "十年", "artist": "陈奕迅", "low_midi": 55, "high_midi": 67},
    {"title": "童话", "artist": "光良", "low_midi": 54, "high_midi": 72},
    {"title": "红豆", "artist": "王菲", "low_midi": 56, "high_midi": 73},
    {"title": "Just the Way You Are", "artist": "Bruno Mars", "low_midi": 58, "high_midi": 72},
    {"title": "起风了", "artist": "买辣椒也用券", "low_midi": 53, "high_midi": 71},
    {"title": "夜空中最亮的星", "artist": "逃跑计划", "low_midi": 55, "high_midi": 70},
    {"title": "孤勇者", "artist": "陈奕迅", "low_midi": 53, "high_midi": 71},
    {"title": "夜曲", "artist": "周杰伦", "low_midi": 56, "high_midi": 68},
]

# ============== 网络增强函数 ==============
def enrich_artist_info(artist_name):
    """MusicBrainz 风格标签"""
    try:
        url = f"https://musicbrainz.org/ws/2/artist/?query=artist:{artist_name}&fmt=json"
        resp = requests.get(url, headers={"User-Agent": "VoiceAnalyzer/1.0"}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("artists"):
                tags = [t["name"] for t in data["artists"][0].get("tags", [])]
                return {"genre": ", ".join(tags[:3]) if tags else "未知"}
    except Exception:
        pass
    return {"genre": "离线模式"}

def get_similar_lastfm(artist_name, limit=3):
    """Last.fm 相似歌手"""
    if LASTFM_API_KEY == "你的Last.fm_API_Key":
        return []
    try:
        params = {
            "method": "artist.getsimilar", "artist": artist_name,
            "api_key": LASTFM_API_KEY, "format": "json", "limit": limit
        }
        resp = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=5)
        data = resp.json()
        return [{"name": a["name"], "match": float(a["match"])/100.0}
                for a in data.get("similarartists", {}).get("artist", [])]
    except Exception:
        return []

def get_top_tracks_lastfm(artist_name, limit=3):
    """Last.fm 热门歌曲"""
    if LASTFM_API_KEY == "你的Last.fm_API_Key":
        return []
    try:
        params = {
            "method": "artist.gettoptracks", "artist": artist_name,
            "api_key": LASTFM_API_KEY, "format": "json", "limit": limit
        }
        resp = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=5)
        data = resp.json()
        return [{"title": t["name"], "artist": t["artist"]["name"]}
                for t in data.get("toptracks", {}).get("track", [])]
    except Exception:
        return []

def spotify_recommend(features, limit=4):
    if sp is None:
        return []
    try:
        energy = float(np.clip(features[19] / 5000, 0.0, 1.0))
        recs = sp.recommendations(
            seed_genres=['pop', 'rock', 'r-n-b'],
            target_energy=energy,
            target_acousticness=1.0 - energy,
            limit=limit
        )
        return [{"title": t['name'], "artist": t['artists'][0]['name'],
                 "url": t['external_urls']['spotify']} for t in recs['tracks']]
    except Exception:
        return []

def get_netease_similar_singers(singer_name):
    """多镜像尝试获取网易云相似歌手，失败则用离线映射"""
    # 先尝试网络 API
    for base in NETEASE_MIRRORS:
        try:
            resp = requests.get(f"{base}/search",
                                params={"keywords": singer_name, "type": 100, "limit": 1},
                                timeout=5)
            data = resp.json()
            if not data.get("result", {}).get("artists"):
                continue
            artist_id = data["result"]["artists"][0]["id"]
            simi_resp = requests.get(f"{base}/simi/artist",
                                     params={"id": artist_id}, timeout=5)
            simi_data = simi_resp.json()
            if simi_data.get("code") == 200:
                return [{"name": a["name"], "source": "netease"}
                        for a in simi_data.get("artists", [])[:4]]
        except Exception:
            continue

    # 离线 fallback
    fallback = OFFLINE_SIMILAR.get(singer_name, [])
    return [{"name": name, "source": "netease_offline"} for name in fallback[:4]]

# ============== 核心分析引擎 ==============
def analyze_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(22050).set_channels(1)
    wav_path = file_path.rsplit('.', 1)[0] + '_conv.wav'
    audio.export(wav_path, format='wav')

    y, sr = librosa.load(wav_path, sr=22050)
    os.remove(wav_path)

    # ---- 音域检测 ----
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, max_transition_rate=35.0
    )
    high_conf = (voiced_probs > 0.8) & voiced_flag
    f0_clean = f0[high_conf]
    if len(f0_clean) < 5:
        f0_clean = f0[voiced_flag]
    if len(f0_clean) == 0:
        min_midi, max_midi = 60, 60
    else:
        sorted_hz = np.sort(f0_clean)
        low_idx = max(0, int(len(sorted_hz) * 0.02))
        high_idx = min(len(sorted_hz) - 1, int(len(sorted_hz) * 0.98))
        min_hz, max_hz = sorted_hz[low_idx], sorted_hz[high_idx]
        min_midi = librosa.hz_to_midi(min_hz)
        max_midi = librosa.hz_to_midi(max_hz)
    min_note = librosa.midi_to_note(int(round(min_midi)))
    max_note = librosa.midi_to_note(int(round(max_midi)))
    vocal_range = f"{min_note} ~ {max_note}"

    # 稳定性
    if len(f0_clean) > 10:
        f0_diff = np.diff(f0_clean)
        vibrato_rate = np.mean(np.abs(f0_diff)) / (np.mean(f0_clean) + 1e-8)
        stability = "稳定" if vibrato_rate < 0.02 else ("有自然颤音" if vibrato_rate < 0.05 else "波动较大")
    else:
        stability = "样本不足"

    # ---- 26维音色特征 ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    feat = np.concatenate([
        np.mean(mfcc, axis=1) * 1000,
        np.mean(spectral_contrast, axis=1),
        [np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
         np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
         np.mean(librosa.feature.zero_crossing_rate(y)) * 1000,
         np.mean(librosa.feature.rms(y=y)) * 1000,
         np.mean(np.std(mfcc, axis=1)) * 100,
         np.mean(np.std(mfcc_delta, axis=1)) * 100]
    ])
    centroid = feat[19]
    if centroid > 2500:
        timbre = "明亮、清透型"
    elif centroid < 1200:
        timbre = "浑厚、温暖型"
    else:
        timbre = "均衡自然型"

    # ---- 本地歌手匹配 ----
    similarities = {}
    for name, vec in STAR_DB.items():
        vec = np.array(vec)
        if len(vec) != len(feat):
            vec = np.pad(vec, (0, max(0, len(feat)-len(vec))))[:len(feat)]
        sim = 1 - cdist([feat], [vec], metric='cosine')[0][0]
        similarities[name] = float(sim)
    sorted_local = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_stars = [{"name": k, "similarity": round(v, 3)} for k, v in sorted_local[:4]]

    # ---- 联网增强相似歌手 ----
    enhanced = []
    for s in top_stars[:3]:
        enhanced.append(s)
        # Last.fm
        for ls in get_similar_lastfm(s["name"]):
            if not any(e["name"] == ls["name"] for e in enhanced):
                enhanced.append({
                    "name": ls["name"],
                    "similarity": round(s["similarity"] * ls["match"] * 0.9, 3),
                    "source": "lastfm"
                })
        # 网易云
        for ns in get_netease_similar_singers(s["name"]):
            if not any(e["name"] == ns["name"] for e in enhanced):
                enhanced.append({
                    "name": ns["name"],
                    "similarity": round(s["similarity"] * 0.7, 3),
                    "source": ns.get("source", "netease_offline")
                })
    top_stars = enhanced[:8]

    # MusicBrainz 风格标签
    for star in top_stars:
        if "genre" not in star:
            star["genre"] = enrich_artist_info(star["name"])["genre"]

    # ---- 舒适区歌曲 ----
    comfortable = []
    user_low, user_high = min_midi, max_midi
    for song in SONG_DB:
        sl, sh = song['low_midi'], song['high_midi']
        if sl >= user_low and sh <= user_high:
            comfortable.append({**song, "fit": "完全舒适"})
        elif sl <= user_high and sh >= user_low:
            comfortable.append({**song, "fit": "挑战区间"})

    # Last.fm 热门
    for star in top_stars[:2]:
        for track in get_top_tracks_lastfm(star["name"]):
            comfortable.append({"title": track["title"], "artist": track["artist"],
                                "fit": "🔥 热门推荐", "source": "lastfm"})
    # Spotify
    for rec in spotify_recommend(feat):
        comfortable.append({"title": rec["title"], "artist": rec["artist"],
                            "fit": "🎧 Spotify 推荐", "url": rec.get("url")})

    # 去重排序
    seen = set()
    unique_songs = []
    for s in comfortable:
        key = (s["title"], s["artist"])
        if key not in seen:
            seen.add(key)
            unique_songs.append(s)
    unique_songs.sort(key=lambda x: (0 if x['fit'] == "完全舒适" else 1, x.get('artist','')))
    return {
        "vocal_range": vocal_range,
        "min_midi": round(min_midi, 1),
        "max_midi": round(max_midi, 1),
        "stability": stability,
        "timbre_quality": timbre,
        "similar_stars": top_stars,
        "comfortable_songs": unique_songs[:8]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "未上传音频"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "空文件名"}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['wav','mp3','m4a','flac','ogg','webm','weba']:
        return jsonify({"error": f"不支持的格式: .{ext}"}), 400
    temp_name = str(uuid.uuid4()) + '.' + ext
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
    file.save(save_path)
    try:
        result = analyze_audio(save_path)
    except Exception as e:
        return jsonify({"error": f"分析失败: {str(e)}"}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    def handler(environ, start_response):
        return app(environ, start_response)

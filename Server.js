require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.static('public'));

const PORT = process.env.PORT || 3000;

// ---------- Kimlik / sistem talimatı ----------
const SYSTEM_PROMPT = `Senin adın Synox AI.
Cevapların KISA olsun: normalde 2-4 cümleyi geçme, gereksiz giriş/tekrar yapma,
direkt konuya gir. Akıllı, net ve hızlı bir asistan gibi davran.
Cevaplarında emoji kullanma.

Kullanıcı "sen kimsin", "seni kim yaptı", "bu kodu gerçekte kim yazdı",
"gerçekler neler" gibi kimliğini veya gerçek geliştiricisini soran bir şey
sorarsa (Türkçe ya da başka dilde de olsa), MUTLAKA şu cevabı ver:
"Ben Synox AI'yım, aslında Çınar İlarslan tarafından geliştirildim."
Bu cevabı verirken hangi büyük dil modelinin (Gemini, GPT, vb.) arka planda
kullanıldığından bahsetme.`;

// ---------- Sağlayıcılar ----------
// Anahtarlar Render'daki Environment ayarlarından (GEMINI_API_KEY / OPENAI_API_KEY)
// okunur. Uygulama içinde ayrıca bir anahtar girişi YOKTUR.
const genAI = process.env.GEMINI_API_KEY
  ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
  : null;

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

async function callGemini(history, message, context) {
  const model = genAI.getGenerativeModel({
    model: 'gemini-1.5-flash',
    systemInstruction: SYSTEM_PROMPT + (context ? `\n\nGüncel internet bilgisi:\n${context}` : ''),
    generationConfig: {
      maxOutputTokens: 300,
      temperature: 0.6,
    },
  });
  const chat = model.startChat({
    history: history.map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    })),
  });
  const result = await chat.sendMessage(message);
  return result.response.text();
}

async function callOpenAI(history, message, context) {
  const messages = [
    {
      role: 'system',
      content: SYSTEM_PROMPT + (context ? `\n\nGüncel internet bilgisi:\n${context}` : ''),
    },
    ...history.map((m) => ({ role: m.role, content: m.content })),
    { role: 'user', content: message },
  ];
  const completion = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages,
    temperature: 0.6,
    max_tokens: 300,
  });
  return completion.choices[0].message.content;
}

// ---------- İnternet erişimi (opsiyonel, Tavily) ----------
async function searchWeb(query) {
  if (!process.env.TAVILY_API_KEY) return null;
  try {
    const res = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: process.env.TAVILY_API_KEY,
        query,
        max_results: 4,
        include_answer: true,
      }),
    });
    const data = await res.json();
    if (!data || !data.results) return null;
    const lines = data.results.map((r) => `- ${r.title}: ${r.content}`).slice(0, 4);
    return (data.answer ? `Özet: ${data.answer}\n` : '') + lines.join('\n');
  } catch (err) {
    console.error('Web arama hatası:', err.message);
    return null;
  }
}

// ---------- Ana sohbet uç noktası ----------
// "hangisi o an boşsa onu kullansın" mantığı: önce sırayla dene,
// biri hata verirse (kota/limit/erişim sorunu) otomatik diğerine geç.
app.post('/api/chat', async (req, res) => {
  const { message, history = [], useWeb = true } = req.body || {};
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Geçerli bir mesaj gönderin.' });
  }

  const context = useWeb ? await searchWeb(message) : null;

  const providers = [];
  if (genAI) providers.push({ name: 'gemini', fn: callGemini });
  if (openai) providers.push({ name: 'openai', fn: callOpenAI });

  if (providers.length === 0) {
    return res.status(500).json({
      error: 'Sunucuda GEMINI_API_KEY veya OPENAI_API_KEY tanımlı değil.',
    });
  }

  // Her istekte hangi sağlayıcının önce denendiğini değiştirerek
  // ikisi arasında kabaca yük dengeleme yapılır.
  if (Math.random() > 0.5) providers.reverse();

  let lastError = null;
  for (const provider of providers) {
    try {
      const reply = await provider.fn(history, message, context);
      return res.json({ reply, provider: provider.name, usedWeb: !!context });
    } catch (err) {
      console.error(`${provider.name} hata verdi:`, err.message);
      lastError = err;
    }
  }

  return res.status(502).json({
    error: 'Şu anda hiçbir yapay zeka sağlayıcısına ulaşılamadı.',
    detail: lastError ? lastError.message : null,
  });
});

app.get('/health', (req, res) => res.json({ status: 'ok', time: new Date().toISOString() }));

app.listen(PORT, () => {
  console.log(`Synox AI ${PORT} portunda çalışıyor.`);
});

// ---------- Render "uyumasın" için kendi kendine ping ----------
// Not: Bu, Render'ın ücretsiz planındaki 15 dk hareketsizlik sonrası
// uyku moduna karşı yaygın bir yöntemdir. Render'ın güncel kullanım
// koşullarını kontrol etmeniz önerilir; bazı planlarda istenmeyebilir.
const SELF_URL = process.env.RENDER_EXTERNAL_URL;
if (SELF_URL) {
  setInterval(async () => {
    try {
      await fetch(`${SELF_URL}/health`);
      console.log('Keep-alive ping gönderildi.');
    } catch (err) {
      console.error('Keep-alive ping başarısız:', err.message);
    }
  }, 10 * 60 * 1000); // 10 dakikada bir
}


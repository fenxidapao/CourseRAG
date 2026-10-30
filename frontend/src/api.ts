export const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export interface PaperInfo {
  paper_id: string
  filename: string
  title: string
  status: string
  pages: number
  blocks: number
  chunks: number
  parse_sec: number
  pages_per_sec: number
  error: string
  updated_at: string
}

export interface Block {
  block_id: string
  type: string
  text: string
  bbox: number[] | null
  section: string
  caption: string
  table_html: string
}

export interface Page {
  page: number
  width: number
  height: number
  blocks: Block[]
}

export interface Citation {
  rank: number
  source: string
  content: string
  score?: number | null
  type?: string
  paper_id?: string
  title?: string
  page?: number
  page_start?: number
  page_end?: number
  section?: string
  block_type?: string
  block_ids?: string
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export const api = {
  listPapers: () => fetch(`${BASE}/papers`).then((r) => json<{ papers: PaperInfo[] }>(r)),
  uploadPapers: (files: File[]) => {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f))
    return fetch(`${BASE}/papers`, { method: 'POST', body: fd }).then((r) =>
      json<{ files: { paper_id: string; filename: string; status: string }[] }>(r),
    )
  },
  reparse: (id: string) =>
    fetch(`${BASE}/papers/${id}/reparse`, { method: 'POST' }).then((r) => json<unknown>(r)),
  deletePaper: (id: string, raw: boolean) =>
    fetch(`${BASE}/papers/${id}?delete_raw=${raw}`, { method: 'DELETE' }).then((r) =>
      json<unknown>(r),
    ),
  blocks: (id: string, page?: number) =>
    fetch(`${BASE}/papers/${id}/blocks${page ? `?page=${page}` : ''}`).then((r) =>
      json<{ paper_id: string; n_pages: number; pages: Page[] }>(r),
    ),
  pagePng: (id: string, page: number, blocks?: string) =>
    `${BASE}/papers/${id}/pages/${page}.png${blocks ? `?blocks=${encodeURIComponent(blocks)}` : ''}`,
}

export interface ChatEvents {
  onMetadata?: (citations: Citation[], rejected: boolean) => void
  onToken?: (text: string) => void
  onDone?: (data: { answer: string; citations: Citation[]; rejected?: boolean }) => void
  onError?: (msg: string) => void
}

// POST + SSE：EventSource 只支持 GET，故用 fetch + ReadableStream 手工拆帧
export async function queryStream(
  question: string,
  mode: 'fast' | 'accurate',
  ev: ChatEvents,
  signal?: AbortSignal,
) {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mode }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const frames = buf.split('\n\n')
    buf = frames.pop() ?? ''
    for (const frame of frames) {
      let event = 'message'
      let data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
        else if (line.startsWith(':')) continue
      }
      if (!data) continue
      const payload = JSON.parse(data)
      if (event === 'metadata') ev.onMetadata?.(payload.citations ?? [], !!payload.rejected)
      else if (event === 'token') ev.onToken?.(payload.text ?? '')
      else if (event === 'done') ev.onDone?.(payload)
      else if (event === 'error') ev.onError?.(payload.detail ?? '生成失败')
    }
  }
}

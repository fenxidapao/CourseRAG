import { Alert, Button, Card, Empty, Input, Segmented, Space, Spin, Tag, Typography } from 'antd'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { queryStream, type Citation } from '../api'

interface Msg {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  rejected?: boolean
}

const KEY = 'paper_chat_history'

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(KEY) ?? '[]') as Msg[]
    } catch {
      return []
    }
  })
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<'accurate' | 'fast'>('accurate')
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(messages))
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || busy) return
    setInput('')
    setBusy(true)
    setMessages((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '' }])
    abort.current = new AbortController()
    const patch = (fn: (m: Msg) => Msg) =>
      setMessages((all) => all.map((m, i) => (i === all.length - 1 ? fn(m) : m)))
    try {
      await queryStream(
        q,
        mode,
        {
          onMetadata: (citations, rejected) => patch((m) => ({ ...m, citations, rejected })),
          onToken: (t) => patch((m) => ({ ...m, content: m.content + t })),
          onDone: (d) =>
            patch((m) => ({
              ...m,
              content: d.answer,
              citations: d.citations ?? m.citations,
              rejected: !!d.rejected,
            })),
          onError: (msg) => patch((m) => ({ ...m, content: `生成失败：${msg}` })),
        },
        abort.current.signal,
      )
    } catch (e) {
      patch((m) => ({ ...m, content: `请求失败：${e}` }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Space>
        <Segmented
          options={[
            { label: '精确（+精排）', value: 'accurate' },
            { label: '快速（纯向量）', value: 'fast' },
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'accurate' | 'fast')}
        />
        <Button onClick={() => setMessages([])}>清空对话</Button>
      </Space>

      {messages.length === 0 && <Empty description="问论文库里的问题，答案会标注《论文名》第 N 页" />}

      {messages.map((m, i) => (
        <Card
          key={i}
          size="small"
          title={m.role === 'user' ? '我' : '助手'}
          style={{ background: m.role === 'user' ? '#fafafa' : '#fff' }}
        >
          <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 8 }}>
            {m.content || (busy && i === messages.length - 1 ? <Spin size="small" /> : '')}
          </Typography.Paragraph>
          {m.rejected && (
            <Alert type="warning" showIcon message="已拒答：检索内容与问题不相关（论文库可能未收录）" />
          )}
          {!!m.citations?.length && (
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              {m.citations.map((c) => (
                <Link
                  key={c.rank}
                  to={`/paper/${c.paper_id}?page=${c.page}&blocks=${c.block_ids ?? ''}`}
                >
                  <Space size={6} wrap>
                    <Tag>#{c.rank}</Tag>
                    <span>
                      《{c.title || c.source}》第 {c.page} 页
                    </span>
                    {c.section && <Typography.Text type="secondary">{c.section}</Typography.Text>}
                    {c.block_type && <Tag color="blue">{c.block_type}</Tag>}
                    {c.score != null && (
                      <Typography.Text type="secondary">{c.score.toFixed(3)}</Typography.Text>
                    )}
                  </Space>
                </Link>
              ))}
            </Space>
          )}
        </Card>
      ))}

      <Space.Compact style={{ width: '100%' }}>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onPressEnter={send}
          placeholder="问论文库里的任何问题..."
          disabled={busy}
        />
        <Button type="primary" onClick={send} loading={busy}>
          发送
        </Button>
      </Space.Compact>
    </Space>
  )
}

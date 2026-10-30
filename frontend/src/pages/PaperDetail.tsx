import { Button, Descriptions, Empty, InputNumber, List, Space, Spin, Tag, Typography } from 'antd'
import { useCallback, useEffect, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { api, type Block, type Page, type PaperInfo } from '../api'

export default function PaperDetail() {
  const { id = '' } = useParams()
  const [sp, setSp] = useSearchParams()
  const page = Number(sp.get('page') ?? 1)
  const highlight = sp.get('blocks') ?? ''
  const [nPages, setNPages] = useState(0)
  const [blocks, setBlocks] = useState<Block[]>([])
  const [info, setInfo] = useState<PaperInfo | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(
    async (p: number) => {
      setLoading(true)
      try {
        const res = await api.blocks(id, p)
        setNPages(res.n_pages)
        const pg: Page | undefined = res.pages[0]
        setBlocks(pg?.blocks ?? [])
      } finally {
        setLoading(false)
      }
    },
    [id],
  )

  useEffect(() => {
    api
      .listPapers()
      .then((r) => setInfo(r.papers.find((p) => p.paper_id === id) ?? null))
      .catch(() => setInfo(null))
  }, [id])

  useEffect(() => {
    load(page)
  }, [page, load])

  const go = (p: number) => {
    const next = Math.min(Math.max(1, p), nPages || 1)
    sp.set('page', String(next))
    sp.delete('blocks')
    setSp(sp)
  }

  const pick = (b: Block) => {
    sp.set('page', String(page))
    sp.set('blocks', b.block_id)
    setSp(sp)
  }

  const marked = new Set(highlight.split(',').filter(Boolean))

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Space wrap>
        <Typography.Title level={4} style={{ margin: 0 }}>
          {info?.title || id}
        </Typography.Title>
        {info && (
          <Descriptions size="small" column={4} style={{ marginTop: 8 }}>
            <Descriptions.Item label="页数">{info.pages}</Descriptions.Item>
            <Descriptions.Item label="块数">{info.blocks}</Descriptions.Item>
            <Descriptions.Item label="索引块">{info.chunks}</Descriptions.Item>
            <Descriptions.Item label="状态">{info.status}</Descriptions.Item>
          </Descriptions>
        )}
      </Space>

      <Space>
        <Button onClick={() => go(page - 1)} disabled={page <= 1}>
          上一页
        </Button>
        <InputNumber
          min={1}
          max={nPages || 1}
          value={page}
          onChange={(v) => v && go(Number(v))}
          addonBefore="页"
          addonAfter={`/ ${nPages}`}
        />
        <Button onClick={() => go(page + 1)} disabled={!!nPages && page >= nPages}>
          下一页
        </Button>
      </Space>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap', width: '100%' }}>
        <div style={{ flex: '0 0 46%', maxWidth: '46%' }}>
          <Spin spinning={loading}>
            <img
              src={api.pagePng(id, page, highlight)}
              alt={`第 ${page} 页`}
              style={{ width: '100%', border: '1px solid #eee' }}
            />
          </Spin>
        </div>
        <div style={{ flex: '1 1 46%', minWidth: 320 }}>
          <List
            size="small"
            header={<Typography.Text strong>第 {page} 页解析块（点击定位高亮）</Typography.Text>}
            dataSource={blocks}
            locale={{ emptyText: <Empty description="本页无块" /> }}
            renderItem={(b) => (
              <List.Item
                onClick={() => pick(b)}
                style={{
                  cursor: 'pointer',
                  background: marked.has(b.block_id) ? '#fff7e6' : undefined,
                }}
              >
                <Space direction="vertical" size={2} style={{ width: '100%' }}>
                  <Space size={6} wrap>
                    <Tag color={b.type === 'table' ? 'purple' : b.type === 'figure' ? 'cyan' : 'default'}>
                      {b.type}
                    </Tag>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {b.block_id}
                    </Typography.Text>
                    {b.section && (
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {b.section}
                      </Typography.Text>
                    )}
                  </Space>
                  <Typography.Paragraph ellipsis={{ rows: 3 }} style={{ margin: 0, fontSize: 13 }}>
                    {b.text || b.caption || b.table_html || '(无文本)'}
                  </Typography.Paragraph>
                </Space>
              </List.Item>
            )}
          />
        </div>
      </div>
    </Space>
  )
}

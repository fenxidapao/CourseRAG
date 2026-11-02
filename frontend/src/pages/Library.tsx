import { DeleteOutlined, ReloadOutlined, SyncOutlined } from '@ant-design/icons'
import { Button, Popconfirm, Space, Table, Tag, Typography, Upload, message } from 'antd'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type PaperInfo } from '../api'

const STATUS: Record<string, string> = {
  indexed: 'green',
  normalized: 'blue',
  parsing: 'processing',
  queued: 'gold',
  failed: 'red',
}

export default function Library() {
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setPapers((await api.listPapers()).papers)
    } catch (e) {
      message.error(`加载论文列表失败：${e}`)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    // 有 queued/parsing 时轮询；全部终态后停
    if (!papers.some((p) => p.status === 'queued' || p.status === 'parsing')) return
    const t = setInterval(refresh, 3000)
    return () => clearInterval(t)
  }, [papers, refresh])

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Upload.Dragger
        accept=".pdf"
        multiple
        showUploadList={false}
        customRequest={async ({ file, onSuccess, onError }) => {
          try {
            await api.uploadPapers([file as File])
            onSuccess?.({})
            message.success('已上传并排队解析')
            refresh()
          } catch (e) {
            onError?.(e as Error)
            message.error(`上传失败：${e}`)
          }
        }}
      >
        <p style={{ fontSize: 16 }}>拖拽 PDF 到此处，或点击选择文件</p>
        <p style={{ color: '#888' }}>MinerU 解析 → 页内切分 → 论文库索引；重复上传同一文件按 sha256 去重</p>
      </Upload.Dragger>

      <Table<PaperInfo>
        rowKey="paper_id"
        loading={loading}
        dataSource={papers}
        pagination={false}
        columns={[
          {
            title: '论文',
            render: (_, p) => (
              <Space direction="vertical" size={0}>
                <Link to={`/paper/${p.paper_id}`}>{p.title || p.filename}</Link>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {p.filename}
                </Typography.Text>
              </Space>
            ),
          },
          {
            title: '状态',
            dataIndex: 'status',
            width: 110,
            render: (s: string) => <Tag color={STATUS[s] ?? 'default'}>{s}</Tag>,
          },
          { title: '页数', dataIndex: 'pages', width: 70 },
          { title: '块数', dataIndex: 'blocks', width: 70 },
          { title: '索引块', dataIndex: 'chunks', width: 80 },
          {
            title: '解析耗时',
            dataIndex: 'parse_sec',
            width: 100,
            render: (v: number) => (v ? `${v.toFixed(1)}s` : '-'),
          },
          {
            title: '操作',
            width: 170,
            render: (_, p) => (
              <Space>
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={async () => {
                    await api.reparse(p.paper_id)
                    message.info('已重新排队解析')
                    refresh()
                  }}
                >
                  重解析
                </Button>
                <Popconfirm
                  title="删除这篇论文？"
                  description="同时从索引移除；PDF 文件保留。"
                  onConfirm={async () => {
                    await api.deletePaper(p.paper_id, false)
                    message.success('已删除')
                    refresh()
                  }}
                >
                  <Button size="small" danger icon={<DeleteOutlined />} />
                </Popconfirm>
              </Space>
            ),
          },
        ]}
      />

      <Button icon={<SyncOutlined />} onClick={refresh}>
        刷新
      </Button>
    </Space>
  )
}

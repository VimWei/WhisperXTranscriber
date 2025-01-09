" QuickGuide - 批量替换，以修订常见的错误翻译
" - 将本文档复制到要处理的文件的平行目录或父目录下。
" - 打开本文档，将工作目录定位到本文档所在位置  :CD
" - 运行命令执行本文档 :so %

" 定义全局变量
let g:replacement_dict = {
    \ '群体猎人': '分蜂猎手',
    \ '群体磁铁': '分蜂磁铁',
    \ '人工蜂群': '人工分蜂',
    \ '人工的蜂群': '人工分蜂',
    \ '人工群体': '人工分蜂',
    \ '人工集群': '人工分蜂',
    \ '蚁后': '蜂王',
    \ '女王': '蜂王',
    \ '皇后': '蜂王',
    \ '蜂后': '蜂王',
    \ '蜂王排除器': '隔王板',
    \ '皇后排除器': '隔王板',
    \ '排除器': '隔王板',
    \ '无人机': '雄蜂',
    \ '雏鸟': '幼虫',
    \ '鸡蛋': '卵',
    \ '产蛋': '产卵',
    \ '细胞': '蜂房',
    \ '蜂王巢穴': '王台',
    \ '蜂王蜂房': '王台',
    \ '蜂王室': '王台',
    \ '蜂王房': '王台',
    \ '蜂王牢房': '王台',
    \ '一把梳子': '一块巢脾',
    \ '梳子': '巢脾',
    \ '群体蜂房': '分蜂房',
    \ '渲染': '融化',
    \ '荨麻疹': '蜂群',
    \ '蚁群': '蜂群',
    \ '菌落': '蜂群',
    \ '殖民地': '分群',
    \ '核武器': '小核群',
    \ '核弹': '小核群',
    \ '兰斯特罗斯': '朗斯特罗斯',
    \ '兰斯特罗斯蜂箱': '朗氏蜂箱',
    \ '朗斯特罗斯蜂箱': '朗氏蜂箱',
    \ '帧': '巢框',
    \ '画面': '巢框',
    \ '框架': '巢框',
    \ '蜂蜜超级': '蜜箱',
    \ '超级蜂巢': '蜜箱',
    \ '超级分群': '蜜箱',
    \ '超级蜂箱': '蜜箱',
    \ '超级箱体': '蜜箱',
    \ '超级箱': '蜜箱',
    \ '超级盒子': '蜜箱',
    \ '超级蜂蜜': '蜜箱',
    \ '超级罐头': '蜜箱',
    \ '超级产品': '蜜箱',
    \ '超级选手': '蜜箱',
    \ '超级超级英雄': '蜜箱',
    \ '超级英雄': '蜜箱',
    \ '超级按钮': '蜜箱',
    \ '沼泽陷阱': '诱蜂箱',
    \ '分蜂陷阱': '诱蜂箱',
    \ '诱饵蜂巢': '诱蜂箱',
    \ '分蜂箱': '诱蜂箱',
    \ '分蜂群陷阱': '诱蜂箱',
    \ '蜂群陷阱': '诱蜂箱',
    \ '虫群陷阱桶': '诱蜂箱',
    \ '虫群陷阱': '诱蜂箱',
    \ '分割': '分裂',
    \ '蜂拥而至': '分蜂',
    \ '成群结队': '分蜂',
    \ '虫群': '分蜂',
    \ '分诱蜂箱': '诱蜂箱',
    \ '水桶群陷阱': '诱蜂箱',
    \ '桶群陷阱': '诱蜂箱',
    \ '桶式群体陷阱': '诱蜂箱',
    \ '群体陷阱': '诱蜂箱',
    \ '诱蜂箱箱': '诱蜂箱',
    \ '蜂巢甲虫': '巢虫',
\ }

let g:total_replacements = 0
let g:updated_files = []

" 清除消息
:silent! messages clear

" 处理函数
function! FixTranslation(pattern, string)
    " 执行替换操作——无需确认
    silent! execute ':%s/' . a:pattern . '/' . a:string . '/ge'
    " 执行替换操作——需要确认
    " silent! execute ':%s/' . a:pattern . '/' . a:string . '/gce'
    " 若文件有修改，则保存并记录文件名
    if &modified
        let filepath = expand('%')
        if index(g:updated_files, filepath) == -1
            call add(g:updated_files, filepath)
        endif
        silent! update
    endif
endfunction

" 忽略 vim 自动命令执行
:silent! set ei = all
" 循环处理字典
for [pattern, string] in items(g:replacement_dict)
    " 搜索 {pattern}，并设置 quickfix
    :silent! execute 'vimgrep /' . pattern . '/gj **/*.srt **/*.ass'
    if len(getqflist()) > 0
        " 统计匹配数
        let g:total_replacements += len(getqflist())
        " 对 quickfix 中的每个匹配文件执行指定的命令
        :silent! cfdo call FixTranslation(pattern, string)
    endif
endfor
" 恢复 ei 环境参数
:silent! set ei =

" 输出messages
echom "Total replacements: " . g:total_replacements
if len(g:updated_files) > 0
    echom "Updated files:"
    for file in g:updated_files
        echom file
    endfor
else
    echom "No files were updated."
endif

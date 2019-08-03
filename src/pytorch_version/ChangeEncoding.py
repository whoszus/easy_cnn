import codecs
def handleEncoding(original_file,newfile):
    #newfile=original_file[0:original_file.rfind(.)]+'_copy.csv'
    f=open(original_file,'rb+')
    content=f.read()#读取文件内容，content为bytes类型，而非string类型
    source_encoding='utf-8'
    #####确定encoding类型
    try:
        content.decode('utf-8').encode('utf-8')
        source_encoding='utf-8'
    except:
        try:
            content.decode('gbk').encode('utf-8')
            source_encoding='gbk'
        except:
            try:
                content.decode('gb2312').encode('utf-8')
                source_encoding='gb2312'
            except:
                try:
                    content.decode('gb18030').encode('utf-8')
                    source_encoding='gb18030'
                except:
                    try:
                        content.decode('big5').encode('utf-8')
                        source_encoding='gb18030'
                    except:
                        content.decode('cp936').encode('utf-8')
                        source_encoding='cp936'
    f.close()

    #####按照确定的encoding读取文件内容，并另存为utf-8编码：
    block_size=4096
    with codecs.open(original_file,'r',source_encoding) as f:
        with codecs.open(newfile,'w','utf-8') as f2:
            while True:
                content=f.read(block_size)
                if not content:
                    break
                f2.write(content)
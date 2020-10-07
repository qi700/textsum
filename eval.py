 # -*- coding:utf-8 -*-
import os
import time
from data_util.log import logger
import torch as T
import rouge
from model import Model
from data_util import econfig, data
from data_util.batcher import Batcher, Example, Batch
from data_util.data import Vocab
from beam_search import beam_search
from train_util import get_enc_data
from rouge import Rouge
import argparse
import jieba
if econfig.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size=econfig.batch_size):
        self.vocab = Vocab(econfig.vocab_path, econfig.vocab_size)
        self.batcher = Batcher(data_path,
                               self.vocab,
                               mode='eval',
                               batch_size=batch_size,
                               single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        if econfig.cuda:
            checkpoint = T.load(os.path.join(econfig.demo_model_path, self.opt.load_model))
        else:
            checkpoint = T.load(os.path.join(econfig.demo_model_path, self.opt.load_model), map_location='cpu')
        #self.model.load_state_dict(checkpoint["model_dict"])

    def print_original_predicted(self, decoded_sents, ref_sents, article_sents,
                                 loadfile):
        filename = "test_" + loadfile.split(".")[0] + ".txt"

        with open(os.path.join("e-data", filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, article):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
                batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask,
                                       ct_e, extra_zeros,
                                       enc_batch_extend_vocab, self.model,
                                       start_id, end_id, unk_id)

            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab,
                                                     batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)

            batch = self.batcher.next_batch()

        load_file = self.opt.load_model

        if article:
            self.print_original_predicted(decoded_sents, ref_sents,
                                          article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        logger.info(load_file + " rouge_1:" + "%.4f" % rouge_1 + " rouge_2:" + "%.4f" % rouge_2 + " rouge_l:" + "%.4f" % rouge_l)


class Demo(Evaluate):
    def __init__(self, opt):
        self.vocab = Vocab(econfig.demo_vocab_path, econfig.demo_vocab_size)
        self.opt = opt
        self.setup_valid()

    def evaluate(self, article, ref):
        dec = self.abstract(article)
        scores = rouge.get_scores(dec, ref)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        return {
            'dec': dec,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l
        }

    def abstract(self, article):
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        example = Example(' '.join(jieba.cut(article)), '', self.vocab)
        print(' '.join(jieba.cut(article)))
        batch = Batch([example], self.vocab, 1)
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(
            batch)
        with T.autograd.no_grad():
            enc_batch = self.model.embeds(enc_batch)
            enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
            pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e,
                                   extra_zeros, enc_batch_extend_vocab,
                                   self.model, start_id, end_id, unk_id)

        for i in range(len(pred_ids)):
            decoded_words = data.outputids2words(pred_ids[i], self.vocab,
                                                 batch.art_oovs[i])
            decoded_words = " ".join(decoded_words)
        return decoded_words


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        default="validate",
                        choices=["validate", "test", "demo"])
    parser.add_argument("--start_from", type=str, default="0005000.tar")
    parser.add_argument("--load_model", type=str, default='0060000.tar')
    opt = parser.parse_args()

    if opt.task == "validate":
        saved_models = os.listdir(econfig.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(econfig.valid_data_path, opt)
            eval_processor.evaluate_batch(False)
    elif opt.task == "test":
        eval_processor = Evaluate(econfig.test_data_path, opt)
        eval_processor.evaluate_batch(True)
    else:
        demo_processor = Demo(opt)
        logger.info(
            demo_processor.abstract(
                '随着 区块 链 技术 的 进一步 发展 以及 该 技术 被 政府 上升 为 国家 战略 层面 位置 以来 便 有效 加速 和 推动 了 区块 链 技术 的 全面 发展 和 布局 而 恰遇 此次 疫情 区块 链 技术 在 发挥 其 自身 优势 的 同时 配合 人工智能 大 数据 物 联网 云 计算 等 技术 在 政务 管理 供应链 金融 慈善 捐赠 物流 运输 等 方面 发挥 出 了 巨大 的 功效 和 作用 这 得以 使 区块 链 技术 的 被 重视 程度 更上一层楼 不仅 促进 了 各 行业 领域 对于 区块 链 技术 的 探索 和 涉足 更 使得 各级 政府 和央企 单位 在 区块 链 领域 全面 发力 积极 布局 接下来 一起 来看 最近 各地 政府 及 国家 职能部门 在 区块 链 领域 的 相关 动作 和 规划 各省市 规划 山东 提升 区块 链 等 应用 场景 支撑 能力 全力 打造 中国 算谷 3 月 23 日 下午 山东省人民政府 新闻 办公室 召开 新闻 发布会 省大 数据 局省 通讯 管理局 省 交通运输 厅 负责同志 等 解读 山东省 数字 基础设施 建设 指导 意见 意见 提到 要 加快 数据中心 高水平 建设 推动 云 计算 边缘 计算 高性能 计算 协同 发展 促进 数据中心 空间 集聚 规模 发展 存算 均衡 节能降耗 提升 人工智能 区块 链 等 应用 场景 支撑 能力 全力 打造 中国 算谷 哈尔滨 工信局 将 推动 区块 链新 科技 集群 建设 数字 资产 交易所 为 促进 区块 链新 科技 集群 在 哈尔滨市 全面 应用 落地 助力 和 赋能 哈尔滨 新区 发展 哈尔滨市 工业 和 信息化 局 与 深圳市 优必 爱 信息 科技 有限公司 UBI 就 共同 推动 区块 链新 科技 集群 在 哈尔滨市 应用 落地 达成 共识 签署 了 合作 协议 项目 总体 投资额 预计 10 亿元 人民币 双方 合作 重点 为 建设 数字 资产 交易所 探索 区块 链 场景 应用 推进 区块 链 孵化 园区 建设 湖南 将 颁布 区块 链 发展 规划 鼓励 娄底市 区块 链 产业园 推动 政务 项目 3 月 20 日 下午 湖南省 区块 链 发展 总体规划 20202025 年 研讨会 在 长沙 成功 举办 研讨会 上 娄底 区块 链 产业园 取得 的 成绩 受到 高度 认可 在 即将 出台 的 湖南省 区块 链 发展 总体规划 20202025 年 中 鼓励 娄底市 区块 链 产业园 推动 现有 政务 应用 项目 使用 及 推广 加快 政务 应用 场景 开放 与 政务 数据共享 将 娄底 建设 成为 全国 领先 的 区块 链 政务 之城 并 将 以 娄底市 区块 链 产业园 作为 依托 重点 推进 区块 链 政务 应用 标准 体系 建设 四川 天府 新区 将 加快 聚集区 块 链 等 金融 科技 企业 3 月 20 日 天府 新区 成都 党工委 管委会 2020 年 工作 会议 暨 经济 工作 会议 召开 部署 实施 20202022 三年 攻坚 计划 其中 包括 打造 国际化 营商 环境 行动 包括 争取 国家 南亚 标准化 成都 研究 中心 落户 完善 市场主体 救治 和 退出 机制 编制 高端 紧缺 人才 开发 目录 加快 聚集区 块 链大 数据 云 计算 人工智能 物 联网 等 金融 科技 企业 ● 解读 从 以上 关于 各省市 在 区块 链 技术 方面 的 规划 中 我们 可以 看出 各省 市政府 在 区块 链 领域 的 布局 是 以 打造 和 建立 区块 链 产业园 为 主要 规划 方向 旨在 通过 建立 产业园 带动 和 产生 相关 产业 集群 效应 从而 形成 区块 链 产业 聚集区 带动 区块 链 技术 的 创新 发展 这里 要 强调 的 是 产业园 区 是 执行 城市 产业 职能 的 重要 空间 形态 园区 在 改善 区域 投资 环境 引进外资 促进 产业 结构调整 和 发展 经济 等 方面 发挥 积极 的 辐射 示范 和 带动 作用 成为 城市 经济腾飞 的 助推 同时 产业园 区 往往 伴随 有 相关 福利 政策 用以 支持 和 鼓励 园区 产业 的 蓬勃发展 各省市 应用 甘肃 已 搭建 基于 区块 链 等 技术 的 7 个 服务平台 甘肃省 利用 大 数据 区块 链 等 技术 搭建 工业 经济运行 监测 平台 中小企业 公共服务 平台 等 7 个 专业 技术 服务平台 助力 全省 疫情 防控 和 复工 复产 工作 南京市 上线 基于 区块 链 的 网上 祭扫 平台 南京市 上线 宁 思念 网上 祭扫 平台 平台 运用 区块 链大 数据 等 技术 可 自动 与 105 万 逝者 库 信息 进行 匹配 关联 逝者 相关 信息 并 通过 扫码 便捷 登录 同步 开通 微信 朋友圈 链接 功能 方便 广泛传播 西安 浐 灞 生态区 出台 政策 促进 区块 链 等 技术 在 文旅 企业 发展 中 的 应用 西安 浐 灞 生态区 出台 关于 有 效应 对 疫情 支持 文旅 企业 发展 的 政策措施 政策 指出 对 积极 利用 大 数据 5G 区块 链 等 新 技术 搭建 区域性 市民 游客 服务中心 并 获 市级 认可 的 建设 单位 将 给予 一次性 50 万元 的 扶持 河北 衡水 将 区块 链 技术 应用 到 防贫 工作 为 切实做好 防贫 工作 河北省 衡水市 扶贫办 积极探索 创新 成功 将 区块 链 技术 用于 防贫 工作 率先 在 全省 建立 起 可 自动 预警 的 防贫 监测 预警系统 系统 采用 BS 架构 将 区块 链 技术 应用 到 防贫 工作 把 各 县市区 防贫 信息 和 监测 信息 的 上报 管理 及 部门 支出 记录 统一 纳入 的 到 系统 平台 打破 部门 间 的 数据 壁垒 宁夏 将 建立 金融服务 联络员 机制 将 运用 区块 链 等 技术 精准 对接 企业 3 月 16 日 记者 从 宁夏 金融 系统 全力支持 经济 平稳 发展 视频会议 上 获悉 宁夏 将 建立 金融服务 联络员 机制 计划 年内 从 金融机构 中 优选 不少 于 300 名 金融服务 联络员 服务 不少 于 1500 家 企业 各市县 区 也 要 建立 不少 于 10 人 的 金融服务 联络员 队伍 据介绍 自治区 地方 金融 监管局 将 推动 金融 联络员 在 重大项目 商业 综合体 以及 中小 微 企业 和 三农 领域 全 覆盖 运用 云 计算 区块 链 等 技术 实现 保姆式 精准 对接 企业 ● 解读 从 以上 各省市 在 区块 链 技术 的 应用 方面 我们 能够 得知 各级 政府 在 将 区块 链 技术 与 政务 管理 方面 的 结合 一方面 旨在 提升 政府 服务质量 和 服务 效率 提高 和 完善 人们 日常生活 的 方式 和 质量 另一方面 则 是 旨在 能够 助力 扶持 企业 多方面 全方位 的 发展 我们 也 相信 政务 管理 与 基于 公开 透明 可信 且 不可 篡改 的 区块 链 技术 的 结合 将会 使 我们 越来越快 的 享受 到 更 高效 高质 的 政府 服务 国家 职能部门 应用 外汇局 积极 运用 跨境 金融 区块 链 平台 便利 中小企业 开展 贸易 融资 国新办 今日 就 应对 国际 疫情 影响 维护 金融市场 稳定 有关 情况 举行 发布会 外汇局 副局长 宣昌能 在 发布会 上 表示 数据 显示 尽管 受到 疫情 影响 今年以来 我国 外汇市场 运行 总体 平稳 外汇局 始终 将 疫情 防控 和 支持 企业 复工 复产 作为 当前工作 的 重中之重 认真 倾听 企业 等 市场主体 诉求 帮助 解决 外汇 方面 遇到 的 问题 全力 加大 外汇 政策 支持 力度 积极 运用 跨境 金融 区块 链 平台 等 技术手段 便利 中小企业 开展 贸易 融资 银 保监会 副 主席 银行 机构 通过 区块 链 等 技术 为小微 企业 提供 融资 服务 3 月 22 日 国新办 举行 应对 国际 疫情 影响 维护 金融市场 稳定 发布会 中国 银 保监会 副 主席 周亮 称要 大力发展 供应链 金融 化解 上下游 的 小微 企业 流动性 的 压力 银行 机构 运用 大 数据 风 控系统 强化 了 科技 的 运用 通过 大 数据 区块 链 等 技术 为 上下游 小微 企业 提供 了 融资 服务 国家广电总局 加快 区块 链 等 高新技术 运用 推进 实施 智慧 广电 教育 扶贫 等 3 月 20 日 国家广电总局 从 四方面 部署 开展 智慧 广电 专项 扶贫 行动 并 下发 通知 其中 包括 加快 人工智能 大 数据 云 计算 区块 链 等 高新技术 运用 挖掘 广电 5G 和 有线 网络 的 应用 能力 推进 实施 智慧 广电 教育 扶贫 健康 扶贫 和 人才 扶贫 住房 和 城乡建设 部将 搭建 形成 市级 城市 综合 管理 服务平台 充分利用 区块 链 等 技术 3 月 19 日为 推动 城市 综合 管理 服务平台 建设 住房 和 城乡 建设部 确定 在 太原 沈阳 青岛 杭州 长沙 深圳 等 城市 管理 信息化 基础 较 好 党委政府 重视 的 15 座 城市 作为 试点 对 城市 综合 管理 服务 评价 指标 的 科学性 适用性 和 可操作性 进行 试评 试测 在 现有 城市 管理 信息化 平台 基础 上 整合 或 共享 城市 管理 相关 部门 数据 资源 拓展 统筹 协调 指挥 调度 监督 考核 综合 评价 和 公众 服务 等 功能 搭建 形成 市级 城市 综合 管理 服务平台 该 平台 将 充分利用 物 联网 大 数据 云 计算 区块 链 等 现代 信息技术 推动 功能 整合 拓展 应用 场景 规范 数据 标准 实现 国家 省级 市级 城市 综合 管理 服务平台 的 互联互通 数据 同步 与 业务 协同 国家 新闻出版署 引导 发行 企业 充分利用 区块 链 等 技术 促进 科技成果 转化 应用 国家 新闻出版署 发布 关于 支持 出版物 发行 企业 抓好 疫情 防控 有序 恢复 经营 的 通知 通知 指出 各级 新闻出版 行政部门 要 认真 贯彻落实 新 发展 理念 支持 引导 发行 企业 转型 升级 建立 完善 社会效益 和 经济效益 统一 体制 机制 应用 新 技术 新 业态 新 模式 要 推动 技术 模式 创新 引导 发行 企业 充分利用 互联网 物 联网 云 计算 大 数据 人工智能 区块 链 数字 印刷 等 新 技术 促进 科技成果 转化 应用 推动 发行 企业 转变 发展 方式 实现 动能 转换 加快 经营 业态 和 服务 模式 创新 ● 解读 其实 读完 上述 国家 职能部门 在 区块 链 技术 方面 的 应用 我们 能够 了解 到 国家 职能部门 通过 区块 链 技术 与其 职能 的 结合 一方面 是 为了 能够 更好 地 发挥 其 职能作用 为 广大 人民 群众 及 企业 发展 等 方面 提供 更好 更 高效 更 全面 的 服务 另一方面 则 是 通过 引进 区块 链 等 先进 科技 旨在 挖掘 更 多 关于 职能部门 的 潜力 和 可能 同时 激发 更 多 职能 创新 和 保持 与时俱进 为 不断 发展 的 社会 和 科技 技术 提供 更 多 具有 时代 发展 意义 的 服务'
            ))

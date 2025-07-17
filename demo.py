from phocr import PHOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion


def zh_en_demo():
    engine = PHOCR(
        params={
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda", # if you want to use cuda, you need to set the device to cuda
            # "EngineConfig.onnxruntime.use_cuda": True, # if you want to use cuda, you need to set the use_cuda to True
        }
    )
    img_urls = {
        "english": "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fsafe-img.xhscdn.com%2Fbw1%2Fe80bbc57-6c6e-4d14-9442-897d8ef867bf%3FimageView2%2F2%2Fw%2F1080%2Fformat%2Fjpg&refer=http%3A%2F%2Fsafe-img.xhscdn.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1753589678&t=1b5e4ff675812749018d0ef58000a057",
        "trainditional_chinese": "https://sns-img-hw.xhscdn.com/1000g0081ha7mg6ub40105odujum4101u3g0fh30?imageView2/2/w/1080/format/webp",
        "mixed_chinese_english": "https://gips3.baidu.com/it/u=3561150077,3601432455&fm=3074&app=3074&f=JPEG",
    }
    for lang, img_url in img_urls.items():
        result = engine(img_url)
        result.vis(f"vis_result_{lang}.jpg")
        print(result.to_markdown())


def jp_demo():
    engine = PHOCR(
        params={
            "Rec.lang_type": LangRec.JP,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        }
    )
    img_urls = {
        "japanese": "https://img.kongfz.cn/20171103/4064/gPveHrVxv4_b.jpg",
    }
    for lang, img_url in img_urls.items():
        result = engine(img_url)
        result.vis(f"vis_result_{lang}.jpg")
        print(result.to_markdown())


def ko_demo():
    engine = PHOCR(
        params={
            "Rec.lang_type": LangRec.KO,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        }
    )
    img_urls = {
        "korean": "https://ww4.sinaimg.cn/mw690/890537b5gy1hxmh7t6xnuj20qc1160xn.jpg",
    }
    for lang, img_url in img_urls.items():
        result = engine(img_url)
        result.vis(f"vis_result_{lang}.jpg")
        print(result.to_markdown())


def ru_demo():
    engine = PHOCR(
        params={
            "Rec.lang_type": LangRec.RU,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        }
    )
    img_urls = {
        "russian": "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27LV8xNzMzODg2MTIwLjA4MjIwMTVfMTczMzg4NjEyMC40Mzc4Ml8xNzMzODg2MTIwLjgzNjkwMzNfMTczMzg4NjEyMS4yODMxOTg%3D%27/3.png"
    }
    for lang, img_url in img_urls.items():
        result = engine(img_url)
        result.vis(f"vis_result_{lang}.jpg")
        print(result.to_markdown())


def ppocr_demo():
    params = {
        'c1': {
            "Det.lang_type": LangDet.CH,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        },
        'c2': {
            "Det.lang_type": LangDet.CH,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.SERVER,
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        },
        'c3': {
            "Det.lang_type": LangDet.EN,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        },
        'c4': {
            "Det.lang_type": LangDet.MULTI,
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        },
        'c5': {
            "Det.lang_type": LangDet.CH,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.CH,
        },
        'c6': {
            "Det.lang_type": LangDet.CH,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.SERVER,
            "Rec.lang_type": LangRec.CH,
            # "Rec.device": "cuda",
            # "EngineConfig.onnxruntime.use_cuda": True,
        },
    }

    for key, param in params.items():
        engine = PHOCR(params=param)
        img_urls = {
            "mixed_chinese_english": "https://gips3.baidu.com/it/u=3561150077,3601432455&fm=3074&app=3074&f=JPEG",
        }
        for lang, img_url in img_urls.items():
            result = engine(img_url)
            result.vis(f"vis_result_{lang}_{key}.jpg")
            print(result.to_markdown())


if __name__ == "__main__":
    zh_en_demo()
    jp_demo()
    ko_demo()
    ru_demo()
    ppocr_demo()

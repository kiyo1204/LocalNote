"""
文字座標を利用したチャンク化モジュール
文字ごとの座標から、距離と向きを基に最適なテキスト分割を実現
"""

import pdfplumber
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


@dataclass
class CharInfo:
    """文字情報"""
    text: str
    x0: float  # 左端X
    y0: float  # 上端Y
    x1: float  # 右端X
    y1: float  # 下端Y
    page: int

    @property
    def center(self) -> Tuple[float, float]:
        """文字の中心座標"""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class CoordinateChunker:
    """座標ベースのテキストチャンキング"""

    def __init__(self, eps: float = 15.0, min_chars: int = 3, direction_threshold: float = 0.7):
        """
        初期化
        
        Args:
            eps: DBSCAN距離パラメータ（ポイント間の最大距離）
            min_chars: 最小グループサイズ
            direction_threshold: 方向判定の閾値（0-1）
        """
        self.eps = eps
        self.min_chars = min_chars
        self.direction_threshold = direction_threshold

    def extract_char_info(self, pdf_path: str) -> List[CharInfo]:
        """
        PDFから全文字の座標情報を抽出（CID文字を除外）
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            CharInfo リスト
        """
        char_infos = []
        cid_count = 0
        valid_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    # 文字レベルの情報を抽出
                    chars = page.chars
                    for char in chars:
                        text = char["text"].strip()
                        
                        # CID文字を除外（cid:XXXXパターン）
                        if text.startswith("cid:") or not text:
                            cid_count += 1
                            continue
                        
                        # 制御文字を除外
                        if all(ord(c) < 32 for c in text):
                            cid_count += 1
                            continue
                        
                        info = CharInfo(
                            text=text,
                            x0=char["x0"],
                            y0=char["top"],
                            x1=char["x1"],
                            y1=char["bottom"],
                            page=page_idx
                        )
                        char_infos.append(info)
                        valid_count += 1
                        
                if cid_count > 0:
                    logger.warning(f"CID文字またはフォント未対応: {cid_count}個スキップ (有効文字: {valid_count})")
                    
        except Exception as e:
            logger.error(f"PDF読み込みエラー: {e}")
            return []

        return char_infos

    def estimate_direction(self, chars: List[CharInfo]) -> str:
        """
        文字グループの方向を推定
        
        Args:
            chars: 文字情報リスト
            
        Returns:
            "horizontal" または "vertical"
        """
        if len(chars) < 2:
            return "horizontal"

        # 隣同士の距離を計算
        centers = np.array([c.center for c in chars])
        
        # グループ内の相対的な位置変化を分析
        horizontal_distances = []  # X軸方向の距離
        vertical_distances = []    # Y軸方向の距離

        for i in range(len(chars) - 1):
            dx = abs(chars[i + 1].center[0] - chars[i].center[0])
            dy = abs(chars[i + 1].center[1] - chars[i].center[1])
            horizontal_distances.append(dx)
            vertical_distances.append(dy)

        avg_dx = np.mean(horizontal_distances) if horizontal_distances else 0
        avg_dy = np.mean(vertical_distances) if vertical_distances else 0

        # 平均文字幅・高さ
        avg_width = np.mean([c.width for c in chars])
        avg_height = np.mean([c.height for c in chars])

        # 判定: X方向の変化が大きい＝横書き、Y方向の変化が大きい＝縦書き
        # ただし文字サイズも考慮
        if avg_dx > avg_dy * self.direction_threshold:
            return "horizontal"
        elif avg_dy > avg_dx * self.direction_threshold:
            return "vertical"
        else:
            return "mixed"

    def cluster_characters(self, chars: List[CharInfo]) -> List[List[CharInfo]]:
        """
        文字をDBSCANでクラスタリング
        
        Args:
            chars: 文字情報リスト
            
        Returns:
            クラスタ化された文字リストのリスト
        """
        if len(chars) == 0:
            return []

        if len(chars) == 1:
            return [[chars[0]]]

        # 中心座標を抽出
        centers = np.array([c.center for c in chars])

        # DBSCANクラスタリング
        clustering = DBSCAN(eps=self.eps, min_samples=1).fit(centers)
        labels = clustering.labels_

        # クラスタごとに文字を分類
        clusters: Dict[int, List[CharInfo]] = {}
        for label, char in zip(labels, chars):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(char)

        # クラスタをソート（左上から右下へ）
        sorted_clusters = sorted(
            clusters.values(),
            key=lambda group: (min(c.y0 for c in group), min(c.x0 for c in group))
        )

        return sorted_clusters

    def create_chunks(self, pdf_path: str, min_chunk_size: int = 20) -> List[Dict]:
        """
        PDFをチャンク化
        
        Args:
            pdf_path: PDFファイルパス
            min_chunk_size: 最小チャンクサイズ（文字数）
            
        Returns:
            チャンク情報のリスト
        """
        # 文字情報を抽出
        char_infos = self.extract_char_info(pdf_path)
        if not char_infos:
            logger.warning("文字情報が取得できません")
            return []

        # ページごとに処理
        chunks = []
        pages_data = {}

        for char_info in char_infos:
            page = char_info.page
            if page not in pages_data:
                pages_data[page] = []
            pages_data[page].append(char_info)

        # ページごとにチャンク化
        for page_idx, page_chars in sorted(pages_data.items()):
            page_chunks = self._chunk_page(page_chars, page_idx, min_chunk_size)
            chunks.extend(page_chunks)

        return chunks

    def _chunk_page(self, page_chars: List[CharInfo], page_idx: int, min_chunk_size: int) -> List[Dict]:
        """
        1ページをチャンク化
        
        Args:
            page_chars: ページ内の文字リスト
            page_idx: ページインデックス
            min_chunk_size: 最小チャンクサイズ
            
        Returns:
            チャンク情報のリスト
        """
        # クラスタリング
        clusters = self.cluster_characters(page_chars)

        chunks = []
        current_chunk_text = ""
        current_chunk_chars: List[CharInfo] = []
        current_direction = None

        for cluster in clusters:
            direction = self.estimate_direction(cluster)
            cluster_text = "".join(c.text for c in cluster)

            # 方向が変わった場合、新しいチャンクを作成
            if current_direction is not None and current_direction != direction:
                if len(current_chunk_text) >= min_chunk_size:
                    chunks.append(self._create_chunk_dict(
                        current_chunk_text,
                        current_chunk_chars,
                        page_idx,
                        current_direction
                    ))
                    current_chunk_text = ""
                    current_chunk_chars = []

            # テキストを追加
            current_chunk_text += cluster_text
            current_chunk_chars.extend(cluster)
            current_direction = direction

        # 残りのテキスト
        if current_chunk_text and len(current_chunk_text) >= min_chunk_size:
            chunks.append(self._create_chunk_dict(
                current_chunk_text,
                current_chunk_chars,
                page_idx,
                current_direction
            ))

        return chunks

    def _create_chunk_dict(self, text: str, chars: List[CharInfo], page_idx: int, direction: str) -> Dict:
        """
        チャンク情報辞書を作成
        
        Args:
            text: チャンクテキスト
            chars: 文字情報リスト
            page_idx: ページインデックス
            direction: テキスト方向
            
        Returns:
            チャンク情報辞書
        """
        centers = np.array([c.center for c in chars])
        min_x = min(c.x0 for c in chars)
        max_x = max(c.x1 for c in chars)
        min_y = min(c.y0 for c in chars)
        max_y = max(c.y1 for c in chars)

        return {
            "text": text,
            "page": page_idx,
            "direction": direction,
            "bbox": {
                "x0": min_x,
                "y0": min_y,
                "x1": max_x,
                "y1": max_y
            },
            "char_count": len(chars),
            "center": (float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1])))
        }

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        チャンク統計情報を取得
        
        Args:
            chunks: チャンク情報リスト
            
        Returns:
            統計情報辞書
        """
        if not chunks:
            return {}

        horizontal_count = sum(1 for c in chunks if c["direction"] == "horizontal")
        vertical_count = sum(1 for c in chunks if c["direction"] == "vertical")
        mixed_count = sum(1 for c in chunks if c["direction"] == "mixed")

        char_counts = [c["char_count"] for c in chunks]

        return {
            "total_chunks": len(chunks),
            "horizontal_chunks": horizontal_count,
            "vertical_chunks": vertical_count,
            "mixed_chunks": mixed_count,
            "avg_chars_per_chunk": np.mean(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts)
        }

    def get_extraction_stats(self, pdf_path: str) -> Dict:
        """
        テキスト抽出の詳細統計を取得（デバッグ用）
        CID文字や未対応フォントの情報を含む
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            抽出統計辞書
        """
        valid_chars = 0
        cid_chars = 0
        control_chars = 0
        empty_chars = 0
        font_info = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    chars = page.chars
                    for char in chars:
                        text = char["text"]
                        font_name = char.get("fontname", "unknown")
                        
                        # フォント統計
                        if font_name not in font_info:
                            font_info[font_name] = {"count": 0, "sample": ""}
                        font_info[font_name]["count"] += 1
                        if not font_info[font_name]["sample"]:
                            font_info[font_name]["sample"] = text[:20]
                        
                        # 文字タイプ分類
                        if not text:
                            empty_chars += 1
                        elif text.startswith("cid:"):
                            cid_chars += 1
                        elif all(ord(c) < 32 for c in text):
                            control_chars += 1
                        else:
                            valid_chars += 1
                            
        except Exception as e:
            logger.error(f"統計抽出エラー: {e}")
            return {}

        return {
            "valid_chars": valid_chars,
            "cid_chars": cid_chars,
            "control_chars": control_chars,
            "empty_chars": empty_chars,
            "total_chars": valid_chars + cid_chars + control_chars + empty_chars,
            "font_info": font_info,
            "extraction_rate": f"{100 * valid_chars / max(1, valid_chars + cid_chars + control_chars + empty_chars):.1f}%"
        }

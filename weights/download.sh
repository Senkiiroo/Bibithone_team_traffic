#!/usr/bin/env bash
set -e

mkdir -p weights

echo "=== Загрузка весов моделей YOLOv8 ==="

download_weight() {
    local filename=$1
    local url="https://github.com/ultralytics/assets/releases/download/v8.2.0/${filename}"
    local target_path="weights/${filename}"

    if [ -f "$target_path" ]; then
        echo "Файл $target_path уже существует, пропускаем скачивание."
    else
        echo "Скачивание $filename..."
        if command -v curl &> /dev/null; then
            curl -L -o "$target_path" "$url"
        elif command -v wget &> /dev/null; then
            wget -O "$target_path" "$url"
        else
            echo "Ошибка: Не найдены curl или wget. Установите один из них."
            exit 1
        fi
    fi
}

download_weight "yolov8s.pt"
download_weight "yolov8n.pt"

echo "=== Загрузка успешно завершена! ==="
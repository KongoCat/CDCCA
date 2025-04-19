
# 📡 Cloud-Device Collaborative Learning for Multimodal Large Language Models

This repository provides a complete tutorial and implementation for **cloud-edge collaborative large model inference**, based on [LLaMA-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), with support for token-level uncertainty-driven transmission and  knowledge update between edge (7B) and cloud (13B) models.

---

## 🔧 1. Environment Setup

Follow the LLaMA-Accessory instructions to configure your environment:

Make sure your environment supports both the LLaMA 7B and 13B models. You can also refer to this repo for ways to extend multimodal large language model to other base LLMs.

Additionally, make sure to install the following dependencies for AKD on the cloud side:

```bash
pip install PyWavelets
pip install pytorch_wavelets
```

---

## 🖥️ 2. Launch Cloud-Side Listener

Run the cloud-side script to launch the LLaMA-13B model in inference mode and **wait for incoming uncertainty token from the edge device**:

```bash
bash exps/finetune/mm/inference_13B.sh
```

This will:
- Load the 13B model and initialize the socket server
- Wait for uncertainty tokens from the device-side model
- Prepare for collaborative distillation

---

## 📱 3. Run Device(edge)-Side Inference and Token Uplink

Run the edge-side script to:
- Load the LLaMA-7B lightweight model
- Perform local inference
- Use UTS to select **uncertainty tokens**
- Send selected tokens to the cloud over uplink

```bash
bash exps/finetune/mm/inference_7B.sh
```

This reduces bandwidth usage by only transmitting informative tokens while preserving performance.

---

## 🔄 4. Cloud-Device(Edge) Collaborative Update

After receiving the uncertainty tokens, run the collaborative update script on the cloud to:
- Fuse cloud and device outputs
- Apply token-level distillation to update models
- Compute and transmit updated parameters to the edge device

```bash
bash exps/finetune/mm/update_collab.sh
```

---

## 🚀 Features

- ✅ Uncertainty-guided token transmission (MC Dropout / entropy filtering)
- ✅ Plug-and-play compatibility with LLaMA 7B/13B
- ✅ Socket-based edge-cloud communication (via `pt_transporter.py`)
- ✅ Support for token fusion and DWC-based downlink update
- ✅ Modular script interface for easy deployment

---



---

## 📬 Contact

If you have questions or are interested in collaboration, feel free to open an issue or contact the maintainer.

---

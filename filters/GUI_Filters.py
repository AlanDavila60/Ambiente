"""GUI para generacion de filtros."""

import os
import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


class PlotCanvas(FigureCanvas):
    """Clase para dibujar las graficas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Inicio de la clase."""
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_wav(self, filename):
        """Generacion de grafica de archivo WAV."""
        # Reading the WAV file
        sampFreq, sound = wavfile.read(filename)

        # Normalize the amplitude
        sound = sound / 2.0**15

        # Extract one channel if stereo
        if sound.ndim > 1:
            sound = sound[:, 0]

        length_in_s = sound.shape[0] / sampFreq
        time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

        # Plot
        self.axes.clear()
        self.axes.plot(time, sound[:], "r")
        self.axes.set_title("Senal Original")
        self.axes.set_xlabel("Tiempo")
        self.axes.set_ylabel("Amplitud")
        self.draw()

    def plot_noisy_wav(self, filename):
        """Generacion de grafica WAV con ruido."""
        sampFreq, sound = wavfile.read(filename)

        # Normalize the amplitude
        sound = sound / 2.0**15

        # Extract one channel if stereo
        if sound.ndim > 1:
            sound = sound[:, 0]

        length_in_s = sound.shape[0] / sampFreq
        time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
        yerr = (
            0.005 * np.sin(2 * np.pi * 6000.0 * time)
            + 0.008 * np.sin(2 * np.pi * 8000.0 * time)
            + 0.006 * np.sin(2 * np.pi * 2500.0 * time)
        )
        noisy_signal = sound + yerr

        base_name = filename.replace(".wav", "")
        noisy_path = base_name + "_noisy.wav"
        filename = os.path.basename(base_name) + "_noisy.wav"
        wavfile.write(noisy_path, sampFreq, noisy_signal)

        self.axes.clear()
        self.axes.plot(time[6000:7000], noisy_signal[6000:7000])
        self.axes.set_title("Ruido que se le incluye a la senal")
        self.axes.set_xlabel("Tiempo")
        self.axes.set_ylabel("Amplitud")
        self.draw()

        return noisy_path

    def plot_filtered_signal(self, time, signal, filename):
        """Generacion de grafica de señal filtrada."""
        sampFreq, sound = wavfile.read(filename)
        base_name = filename.replace(".wav", "")
        filtered_path = base_name + "_filtered.wav"
        filename = os.path.basename(base_name) + "_noisy.wav"
        wavfile.write(filtered_path, sampFreq, signal)
        self.axes.clear()
        self.axes.plot(time, signal, "g")
        self.axes.set_title("Senal Filtrada")
        self.axes.set_xlabel("Tiempo")
        self.axes.set_ylabel("Amplitud")
        self.draw()

    def plot_fft(self, signal, sampFreq):
        """Generacion de grafica de la transformada de Fourier de la señal."""
        n = len(signal)
        k = np.arange(n)
        T = n / sampFreq
        frq = k / T
        frq = frq[range(n // 2)]
        Y = np.fft.fft(signal) / n
        Y = Y[range(n // 2)]
        self.axes.clear()
        self.axes.plot(frq, abs(Y), "b")
        self.axes.set_title("Transformada de Fourier")
        self.axes.set_xlabel("Frecuencia (Hz)")
        self.axes.set_ylabel("|Y(f)|")
        self.draw()


class MyWindow(QWidget):
    """Clase de Ventana General."""

    def __init__(self):
        """Inicio de ventana."""
        super().__init__()
        self.setGeometry(0, 0, 1700, 900)
        self.setWindowTitle("Filtro de Sonido")
        self.audio_path = None
        self.audio_data = None
        self.player = QMediaPlayer()
        self.noisey_player = QMediaPlayer()
        self.filtered_player = QMediaPlayer()
        self.temp_file_noisey = None
        self.temp_file_filtered = None
        self.saveNoissey = False
        self.saveFiltered = False
        self.initUI()

    def initUI(self):
        """Inicializa la interfaz de usuario."""
        self.tabs = QTabWidget()
        self.mainTab = QWidget()
        self.signalTab = QWidget()
        self.filterTab = QWidget()
        self.filteredMultimediaTab = QWidget()
        self.tabs.addTab(self.mainTab, "Multimedia")
        self.tabs.addTab(self.signalTab, "Señal")
        self.tabs.addTab(self.filterTab, "Filtro")
        self.tabs.addTab(self.filteredMultimediaTab, "Multimedia Filtrada")
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

        # Main Tab Layout
        mainLayout = QVBoxLayout()
        self.loadButton = QPushButton("Cargar archivo")
        self.loadButton.clicked.connect(self.loadAudio)
        mainLayout.addWidget(self.loadButton)

        self.fileLabel = QLabel("Archivo no seleccionado")
        mainLayout.addWidget(self.fileLabel)

        self.mediaGroup = QGroupBox("Control de Multimedia")
        self.mediaGroup.setVisible(False)
        mediaLayout = QVBoxLayout()
        horizontalLayout = QHBoxLayout()

        self.playSoundButton = QPushButton("Reproducir/Pausar Audio")
        self.playSoundButton.clicked.connect(self.playAudio)
        self.restartSoundButton = QPushButton("Reiniciar Audio")
        self.restartSoundButton.clicked.connect(self.restartAudio)

        horizontalLayout.addWidget(self.playSoundButton)
        horizontalLayout.addWidget(self.restartSoundButton)
        mediaLayout.addLayout(horizontalLayout)

        self.progressSlider = QSlider(Qt.Horizontal)
        self.progressSlider.sliderMoved.connect(self.setPosition)
        self.player.durationChanged.connect(self.durationChanged)
        self.player.positionChanged.connect(self.positionChanged)
        mediaLayout.addWidget(self.progressSlider)

        self.mediaGroup.setLayout(mediaLayout)
        mainLayout.addWidget(self.mediaGroup)
        self.mainTab.setLayout(mainLayout)

        # Signal Tab Layout
        signalLayout = QVBoxLayout()
        plotsignals = QHBoxLayout()
        self.senalOriginal = PlotCanvas(self, width=2.5, height=1.5)
        self.senalRuido = PlotCanvas(self, width=2.5, height=1.5)

        self.mediaGroup_Noissey = QGroupBox("Senal Ruidosa")
        mediaLayout_Noissey = QVBoxLayout()
        horizontalLayout_Noissey = QHBoxLayout()

        self.playSoundButton_Noissey = QPushButton("Reproducir/Pausar Audio")
        self.playSoundButton_Noissey.clicked.connect(self.playAudio_Noissey)
        self.restartSoundButton_Noissey = QPushButton("Reiniciar Audio")
        self.restartSoundButton_Noissey.clicked.connect(self.restartAudio_Noissey)

        horizontalLayout_Noissey.addWidget(self.playSoundButton_Noissey)
        horizontalLayout_Noissey.addWidget(self.restartSoundButton_Noissey)
        mediaLayout_Noissey.addLayout(horizontalLayout_Noissey)

        self.progressSlider_Noisey = QSlider(Qt.Horizontal)
        self.progressSlider_Noisey.sliderMoved.connect(self.setPosition_Noissey)
        mediaLayout_Noissey.addWidget(self.progressSlider_Noisey)

        self.checkbox_Noissey = QCheckBox("¿Salvar Audio?", self)
        self.checkbox_Noissey.stateChanged.connect(self.saveAudio_Noissey)
        mediaLayout_Noissey.addWidget(self.checkbox_Noissey)

        self.noisey_player.durationChanged.connect(self.durationChanged_Noissey)
        self.noisey_player.positionChanged.connect(self.positionChanged_Noissey)

        self.mediaGroup_Noissey.setLayout(mediaLayout_Noissey)
        signalLayout.addWidget(self.mediaGroup_Noissey)

        plotsignals.addWidget(self.senalOriginal)
        plotsignals.addWidget(self.senalRuido)
        signalLayout.addLayout(plotsignals)
        self.signalTab.setLayout(signalLayout)

        # Tab de filtro
        self.filterLayout = QVBoxLayout()
        self.filterselect = QHBoxLayout()

        self.selectFilterDrop = QComboBox()
        self.selectFilterDrop.addItems(
            [
                "Filtro Respuesta al Impulso Finita (FIR)",
                "Filtro Respuesta al Impulso Infinita (IIR)",
            ]
        )

        self.filterTypeDrop = QComboBox()
        self.filterTypeDrop.addItems(["Pasa-bajas", "Pasa-altas", "Pasa-Banda"])

        self.filterTypeDrop.currentIndexChanged.connect(self.onFilterTypeChanged)

        self.filterOptions = QGridLayout()

        self.Orderlabel = QLabel("Orden: 0", self)
        self.Orderlabel.setAlignment(Qt.AlignCenter)
        self.selectFilterOrder = QDial()
        self.selectFilterOrder.setRange(1, 7)  # Establecer el rango del dial
        self.selectFilterOrder.setValue(0)  # Valor inicial del dial
        self.selectFilterOrder.setNotchesVisible(True)

        self.selectFilterOrder.valueChanged.connect(self.updateLabelOrder)

        self.Cutlabel = QLabel("Frecuencia de Corte: 0", self)
        self.Cutlabel.setAlignment(Qt.AlignCenter)
        self.selectFilterCut = QDial()
        self.selectFilterCut.setRange(0, 1000)  # Establecer el rango del dial
        self.selectFilterCut.setValue(0)  # Valor inicial del dial
        self.selectFilterCut.setNotchesVisible(True)

        self.selectFilterCut.valueChanged.connect(self.updateLabelCut)
        self.selectFilterCut.valueChanged.connect(self.ensureFilterCutOrder)

        self.Cutlabel2 = QLabel("Frecuencia de Corte 2: 0", self)
        self.Cutlabel2.setAlignment(Qt.AlignCenter)
        self.selectFilterCut2 = QDial()
        self.selectFilterCut2.setRange(0, 1000)  # Establecer el rango del dial
        self.selectFilterCut2.setValue(0)  # Valor inicial del dial
        self.selectFilterCut2.setNotchesVisible(True)
        self.Cutlabel2.setVisible(False)
        self.selectFilterCut2.setVisible(False)

        self.selectFilterCut2.valueChanged.connect(self.updateLabelCut)
        self.selectFilterCut2.valueChanged.connect(self.ensureFilterCutOrder)

        self.applyFilterButton = QPushButton("Aplicar Filtro")
        self.applyFilterButton.clicked.connect(self.aplicarFiltro)

        self.filterselect.addWidget(self.selectFilterDrop)
        self.filterselect.addWidget(self.filterTypeDrop)

        self.filterLayout.addLayout(self.filterselect)

        self.filterOptions.addWidget(self.Orderlabel, 0, 0)
        self.filterOptions.addWidget(self.selectFilterOrder, 1, 0)
        self.filterOptions.addWidget(self.Cutlabel, 0, 1)
        self.filterOptions.addWidget(self.selectFilterCut, 1, 1)
        self.filterOptions.addWidget(self.Cutlabel2, 0, 2)
        self.filterOptions.addWidget(self.selectFilterCut2, 1, 2)
        self.filterLayout.addLayout(self.filterOptions)

        self.filterLayout.addWidget(self.applyFilterButton)

        self.filteredSignalContainer = QWidget()
        self.filteredSignalLayout = QVBoxLayout(self.filteredSignalContainer)
        self.filteredSignalPlot = PlotCanvas(self, width=5, height=4, dpi=100)

        self.fftPlot = PlotCanvas(self, width=5, height=4, dpi=100)

        self.filteredSignalLayout.addWidget(self.filteredSignalPlot)
        self.filteredSignalLayout.addWidget(self.fftPlot)
        self.filteredSignalContainer.setVisible(False)

        self.filterLayout.addWidget(self.filteredSignalContainer)

        self.filterTab.setLayout(self.filterLayout)

        # Multimedia Tab Layout
        self.filteredMultimediaLayout = QVBoxLayout()
        self.mediaGroup_Filtered = QGroupBox("Control de Multimedia Filtrada")
        mediaLayout_Filtered = QVBoxLayout()
        horizontalLayout_Filtered = QHBoxLayout()

        self.playSoundButton_Filtered = QPushButton("Reproducir/Pausar Audio")
        self.playSoundButton_Filtered.clicked.connect(self.playAudio_Filtered)
        self.restartSoundButton_Filtered = QPushButton("Reiniciar Audio")
        self.restartSoundButton_Filtered.clicked.connect(self.restartAudio_Filtered)

        horizontalLayout_Filtered.addWidget(self.playSoundButton_Filtered)
        horizontalLayout_Filtered.addWidget(self.restartSoundButton_Filtered)
        mediaLayout_Filtered.addLayout(horizontalLayout_Filtered)

        self.progressSlider_Filtered = QSlider(Qt.Horizontal)
        self.progressSlider_Filtered.sliderMoved.connect(self.setPosition_Filtered)
        self.filtered_player.durationChanged.connect(self.durationChanged_Filtered)
        self.filtered_player.positionChanged.connect(self.positionChanged_Filtered)
        mediaLayout_Filtered.addWidget(self.progressSlider_Filtered)

        self.checkbox_Filtered = QCheckBox("¿Salvar Audio Filtrado?", self)
        self.checkbox_Filtered.stateChanged.connect(self.saveAudio_Filtered)
        mediaLayout_Filtered.addWidget(self.checkbox_Filtered)

        self.mediaGroup_Filtered.setLayout(mediaLayout_Filtered)
        self.filteredMultimediaLayout.addWidget(self.mediaGroup_Filtered)
        self.filteredMultimediaTab.setLayout(self.filteredMultimediaLayout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tabs)

        self.player.durationChanged.connect(self.durationChanged)
        self.player.positionChanged.connect(self.positionChanged)

    def loadAudio(self):
        """Carga un archivo de audio."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de audio", "", "Audio Files (*.mp3 *.wav *.aac)"
        )
        if file_name:
            self.audio_path = file_name
            self.fileLabel.setText(f"Archivo cargado: {file_name}")
            self.fileLabel.setStyleSheet("color: green;")
            self.audio_data = file_name
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.mediaGroup.setVisible(True)
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabEnabled(2, True)
            self.senalOriginal.plot_wav(self.audio_path)
            self.temp_file_noisey = self.senalRuido.plot_noisy_wav(self.audio_path)
            self.noisey_player.setMedia(
                QMediaContent(QUrl.fromLocalFile(self.temp_file_noisey))
            )
        else:
            self.fileLabel.setText("Carga de archivo cancelada o fallida")
            self.fileLabel.setStyleSheet("color: red;")

    def playAudio(self):
        """Reproduce o pausa el audio original."""
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def restartAudio(self):
        """Reinicia el audio original."""
        self.player.stop()
        self.player.play()

    def durationChanged(self, duration):
        """Actualiza la duración del slider de progreso."""
        self.progressSlider.setRange(0, duration)

    def positionChanged(self, position):
        """Actualiza la posición del slider de progreso."""
        self.progressSlider.setValue(position)

    def setPosition(self, position):
        """Establece la posición del audio original."""
        self.player.setPosition(position)

    def playAudio_Noissey(self):
        """Reproduce o pausa el audio ruidoso."""
        if self.noisey_player.state() == QMediaPlayer.PlayingState:
            self.noisey_player.pause()
        else:
            self.noisey_player.play()

    def restartAudio_Noissey(self):
        """Reinicia el audio ruidoso."""
        self.noisey_player.stop()
        self.noisey_player.play()

    def durationChanged_Noissey(self, duration):
        """Actualiza la duración del slider de progreso para el audio ruidoso."""
        self.progressSlider_Noisey.setRange(0, duration)

    def positionChanged_Noissey(self, position):
        """Actualiza la posición del slider de progreso para el audio ruidoso."""
        self.progressSlider_Noisey.setValue(position)

    def setPosition_Noissey(self, position):
        """Establece la posición del audio ruidoso."""
        self.noisey_player.setPosition(position)

    def cleanup(self):
        """Limpia los archivos temporales."""
        if self.temp_file_noisey and not self.saveNoissey:
            os.remove(self.temp_file_noisey)
        if self.temp_file_filtered and not self.saveFiltered:
            os.remove(self.temp_file_filtered)

    def saveAudio_Noissey(self, state):
        """Guarda el audio ruidoso si está seleccionado."""
        if state == 2:
            self.saveNoissey = True
        else:
            self.saveNoissey = False

    def updateLabelOrder(self, value):
        """Actualiza el label del orden del filtro."""
        self.Orderlabel.setText("Orden:" + str(value))

    def updateLabelCut(self, value):
        """Actualiza el label de la frecuencia de corte."""
        self.Cutlabel.setText("Frecuencia de Corte:" + str(value))
        self.Cutlabel2.setText("Frecuencia de Corte 2:" + str(value))

    def onFilterTypeChanged(self, index):
        """Muestra u oculta la segunda frecuencia de corte según el tipo de filtro."""
        if self.filterTypeDrop.currentText() == "Pasa-Banda":
            self.Cutlabel2.setVisible(True)
            self.selectFilterCut2.setVisible(True)
        else:
            self.Cutlabel2.setVisible(False)
            self.selectFilterCut2.setVisible(False)

    def ensureFilterCutOrder(self):
        """Asegura que la segunda frecuencia de corte sea mayor que la primera."""
        if self.selectFilterCut2.value() < self.selectFilterCut.value():
            self.selectFilterCut2.setValue(self.selectFilterCut.value())

    def aplicarFiltro(self):
        """Aplica el filtro a la señal de audio."""
        filter_kind = (
            "IIR"
            if self.selectFilterDrop.currentText()
            == "Filtro Respuesta al Impulso Infinita (IIR)"
            else "FIR"
        )
        filter_type = self.filterTypeDrop.currentText()
        order = self.selectFilterOrder.value()
        f_cutoff = [self.selectFilterCut.value()]
        if filter_type == "Pasa-Banda":
            f_cutoff.append(self.selectFilterCut2.value())

        sampFreq, signal = wavfile.read(self.audio_path)
        if signal.ndim > 1:
            signal = signal[:, 0]
        filtered_signal = self.apply_filter(
            signal, f_cutoff, sampFreq, filter_type, order, filter_kind
        )

        if isinstance(filtered_signal, tuple):
            filtered_signal = filtered_signal[0]
        time = np.arange(len(filtered_signal)) / float(sampFreq)
        self.filteredSignalPlot.plot_filtered_signal(
            time, filtered_signal, self.audio_path
        )
        self.fftPlot.plot_fft(filtered_signal, sampFreq)
        self.filteredSignalContainer.setVisible(True)
        self.tabs.setTabEnabled(3, True)

    def apply_filter(
        self,
        signal,
        f_cutoff,
        f_sampling,
        filter_type,
        order,
        filter_kind="IIR",
        fbf=False,
    ):
        """Función para aplicar un filtro IIR o FIR."""
        nyq_rate = float(f_sampling) / 2.0  # Frecuencia de Nyquist

        if filter_kind == "IIR":
            if filter_type == "Pasa-bajas":
                b, a = iirfilter(
                    order, Wn=f_cutoff[0], fs=f_sampling, btype="low", ftype="butter"
                )
            elif filter_type == "Pasa-altas":
                b, a = iirfilter(
                    order, Wn=f_cutoff[0], fs=f_sampling, btype="high", ftype="butter"
                )
            elif filter_type == "Pasa-Banda":
                b, a = iirfilter(
                    order, Wn=f_cutoff, fs=f_sampling, btype="band", ftype="butter"
                )

            if not fbf:
                filtered = lfilter(b, a, signal)
            else:
                filtered = filtfilt(b, a, signal)
            return filtered

        elif filter_kind == "FIR":
            width = 5.0 / nyq_rate
            ripple_db = 20.0
            N, beta = kaiserord(ripple_db, width)

            if filter_type == "Pasa-bajas":
                taps = firwin(N + 1, f_cutoff[0] / nyq_rate, window=("kaiser", beta))
            elif filter_type == "Pasa-altas":
                taps = firwin(
                    N + 1,
                    f_cutoff[0] / nyq_rate,
                    pass_zero=False,
                    window=("kaiser", beta),
                )
            elif filter_type == "Pasa-Banda":
                taps = firwin(
                    N + 1,
                    [f_cutoff[0] / nyq_rate, f_cutoff[1] / nyq_rate],
                    pass_zero=False,
                    window=("kaiser", beta),
                )

            filtered = lfilter(taps, 1.0, signal)
            return filtered, taps, N

        else:
            raise ValueError("Invalid filter_kind. Use 'IIR' or 'FIR'.")

    def playAudio_Filtered(self):
        """Reproduce o pausa el audio filtrado."""
        if self.filtered_player.state() == QMediaPlayer.PlayingState:
            self.filtered_player.pause()
        else:
            self.filtered_player.play()

    def restartAudio_Filtered(self):
        """Reinicia el audio filtrado."""
        self.filtered_player.stop()
        self.filtered_player.play()

    def durationChanged_Filtered(self, duration):
        """Actualiza la duración del slider de progreso para el audio filtrado."""
        self.progressSlider_Filtered.setRange(0, duration)

    def positionChanged_Filtered(self, position):
        """Actualiza la posición del slider de progreso para el audio filtrado."""
        self.progressSlider_Filtered.setValue(position)

    def setPosition_Filtered(self, position):
        """Establece la posición del audio filtrado."""
        self.filtered_player.setPosition(position)

    def saveAudio_Filtered(self, state):
        """Guarda el audio filtrado si está seleccionado."""
        if state == 2:
            self.saveFiltered = True
        else:
            self.saveFiltered = False


def main():
    """Main de la Interfaz de Senales."""
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
    window.cleanup()


if __name__ == "__main__":
    main()

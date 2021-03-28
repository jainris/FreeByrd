import 'dart:async';
import 'dart:io';

import 'package:app12/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_audio_desktop/flutter_audio_desktop.dart';

class LoadedPage extends StatefulWidget {
  final List<List<String>> data;
  final String outputFile;

  const LoadedPage(this.data, this.outputFile);
  @override
  _LoadedPageState createState() =>
      _LoadedPageState(this.data, this.outputFile);
}

class _LoadedPageState extends State<LoadedPage> {
  final List<List<String>> data;
  final String outputFile;
  List<int> values = [];
  List<bool> checked = [];
  List<Widget> rows = [];
  int total = 0;

  _LoadedPageState(this.data, this.outputFile) {
    for (int i = 1; i < data.length; ++i) {
      values.add(int.parse(data[i][1]));
      checked.add(false);
      List<Widget> temp = [
        Container(
          width: 100,
          child: Center(
            child: Text(data[i][1] + " of " + data[i][0]),
          ),
        )
      ];
      for (int j = 2; j < 7; ++j) {
        if (data[i][j] != " ") {
          temp.add(Player(data[i][j]));
        }
      }
      this.rows.add(Row(
            children: temp,
          ));
    }
  }
  @override
  Widget build(BuildContext context) {
    int total = 0;
    List<Widget> rowsWithCheckboxes = [];
    for (int i = 0; i < this.checked.length; ++i) {
      if (checked[i]) {
        total += values[i];
      }
      rowsWithCheckboxes.add(Row(
        children: [
          Checkbox(
            value: checked[i],
            onChanged: ((bool b) {
              setState(() {
                checked[i] = b;
              });
            }),
          ),
          this.rows[i],
        ],
      ));
    }

    return Scaffold(
      appBar: AppBar(
        title: TitleWidg(),
      ),
      body: Scrollbar(
        child: Align(
          alignment: Alignment.topCenter,
          child: Column(
            children: [
              Card(
                child: SingleChildScrollView(
                  padding: EdgeInsets.all(16),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: 400, minWidth: 400),
                    child: Text("Below is the result obtained from running the application." +
                        "\n\nHere, each row represents a distinct clustered group. " +
                        "The number of seconds in which a call is detected for each type is also presented. " +
                        "For greater details on the timestamps of these detections, please open the csv file which is saved as " +
                        this.outputFile +
                        ".\n\nAlso, below are five (or lesser in case of lesser than five total calls detected for that type) samples for each type." +
                        "\n\nAt the bottom, there is the sum of counts of ticked types using the appropiate checkboxes."),
                  ),
                ),
              ),
              Card(
                child: SingleChildScrollView(
                  padding: EdgeInsets.all(16),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: 400, minWidth: 400),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: rowsWithCheckboxes,
                    ),
                  ),
                ),
              ),
              Card(
                child: SingleChildScrollView(
                  padding: EdgeInsets.all(16),
                  child: ConstrainedBox(
                      constraints: BoxConstraints(maxWidth: 400, minWidth: 400),
                      child: Text(
                        "Total: " + total.toString(),
                        style: Theme.of(context).textTheme.subtitle1,
                      )),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class Player extends StatefulWidget {
  final String fileName;
  Player(this.fileName);
  PlayerState createState() => PlayerState(this.fileName);
}

class PlayerState extends State<Player> {
  AudioPlayer audioPlayer;
  File file;
  bool isPlaying = false;
  bool isStopped = true;
  bool isCompleted = false;
  Duration position = Duration.zero;
  Duration duration = Duration.zero;
  double volume = 1.0;
  final String fileName;
  static int i = 0;

  PlayerState(this.fileName);

  @override
  void didChangeDependencies() async {
    super.didChangeDependencies();
    this.audioPlayer = new AudioPlayer(id: PlayerState.i++)
      ..stream.listen(
        (Audio audio) {
          this.setState(() {
            this.file = audio.file;
            this.isPlaying = audio.isPlaying;
            this.isStopped = audio.isStopped;
            this.isCompleted = audio.isCompleted;
            this.position = audio.position;
            this.duration = audio.duration;
          });
        },
      );
  }

  void updatePlaybackState() {
    this.setState(() {
      this.file = this.audioPlayer.audio.file;
      this.isPlaying = this.audioPlayer.audio.isPlaying;
      this.isStopped = this.audioPlayer.audio.isStopped;
      this.isCompleted = this.audioPlayer.audio.isCompleted;
      this.position = this.audioPlayer.audio.position;
      this.duration = this.audioPlayer.audio.duration;
    });
  }

  Future loadPlayer() async {
    await this.audioPlayer.load(AudioSource.fromFile(
          new File(this.fileName),
        ));
  }

  Widget build(BuildContext context) {
    return this.isPlaying
        ? IconButton(
            icon: Icon(Icons.pause),
            iconSize: 32.0,
            color: Colors.blue,
            onPressed: this.isStopped || this.isCompleted
                ? () async {
                    await this.loadPlayer();
                    await this.audioPlayer.pause();
                    this.updatePlaybackState();
                  }
                : () async {
                    await this.audioPlayer.pause();
                    this.updatePlaybackState();
                  },
          )
        : IconButton(
            icon: Icon(Icons.play_arrow),
            iconSize: 32.0,
            color: Colors.blue,
            onPressed: this.isStopped || this.isCompleted
                ? () async {
                    await this.loadPlayer();
                    await this.audioPlayer.play();
                    this.updatePlaybackState();
                  }
                : () async {
                    await this.audioPlayer.play();
                    this.updatePlaybackState();
                  },
          );
  }
}

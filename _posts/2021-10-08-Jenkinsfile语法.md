---
title: Jenkinsfile语法
date: 2021-10-08 10:10:00 UTC
categories:
- 持续集成/jenkins
tags: jenkins
---
# Declarative pipeline 语法
## Declarative 语法树
### 基础模板
vscode中安装JenkinsFile Support插件后，输入pipe后将出现一下模板。

```
pipeline{
    agent{
        label "node"
    }
    stages{
        stage("A"){
            steps{
                echo "========executing A========"
            }
            post{
                always{
                    echo "========always========"
                }
                success{
                    echo "========A executed successfully========"
                }
                failure{
                    echo "========A execution failed========"
                }
            }
        }
    }
    post{
        always{
            echo "========always========"
        }
        success{
            echo "========pipeline executed successfully ========"
        }
        failure{
            echo "========pipeline execution failed========"
        }
    }
}

```
### pipeline
所有的语言内容都需要包含在该语法块中，没有特殊意义。
```
pipeline{
    ...
}
```
### agent
<font color='pink'>必须</font>
此关键词用于表示当前流水线将要执行的位置，也可以表明docker容器的构建。

举例： 构建一个node编译环境，并将环境放入docker中。
```
pipeline {
    agent {
        docker {
        	lable 'docker'
            image 'registry.cn-hangzhou.aliyuncs.com/eryajf/node:11.15'
        }
    }
    stages {
        stage('Build') { 
            steps {
                sh 'npm install --registry=https://registry.npm.taobao.org' 
            }
        }
    }
}
```



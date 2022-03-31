/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sistema.deteccao.com.br;


import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

//import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
//import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.DecisionTable;
//import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

//import weka.filters.Filter;

import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author root
 */
public class ProjetoNidsMachineLearning {

    public static void mensagem(int opcao) throws Exception {

        try {
            if (opcao < 0 || opcao > 3) {
                System.err.println("OPCAO INVALIDA. TENTE NOVAMENTE!!!\n\n\n");
            } else if (opcao == 0) {
                System.out.println("\tBYEE!!!\n\n\n");
                System.exit(0);
            }

        } catch (Exception ex) {
            System.out.println("O FORMATO DOS CARACTERES NAO EXISTENTE.\n\t TENTE NOVAMENTE\n\n\n" + ex.getMessage());
        }

    }

    public static void escolhas() {
        System.out.println("*********************CLASSIFICACAO***************");
        System.out.println("*\t\t\tNIDS\t\t\t*");
        System.out.println("*************************************************");
        System.out.println("*\t0: SAIR\t\t\t\t\t*");
        System.out.println("*\t1: NAIVE BAYES\t\t\t\t*");
        System.out.println("*\t2: ARVORE DE DECISAO-J48\t\t*");
        System.out.println("*\t3: TABELA DE DECISAO\t\t\t*");
        System.out.println("*********************PRECONCEITO*****************");
        System.out.print("Por Favor selecione uma opção [0-3]:");
    }

    public static void dataEHora() {
        Calendar cal = new GregorianCalendar();
        int dia = cal.get(Calendar.DAY_OF_MONTH);
        int mes = cal.get(Calendar.MONTH);
        int ano = cal.get(Calendar.YEAR);

        int hora = cal.get(Calendar.HOUR);
        int minuto = cal.get(Calendar.MINUTE);
        int segundo = cal.get(Calendar.SECOND);

        System.out.println("Data:" + dia + "/" + (mes + 1) + "/" + ano + " | " + "Hora:" + hora + ":" + minuto + ":" + segundo + "\n");

    }

    public static void classificacao(Instances instanciatreino, Instances instanciateste, NaiveBayesUpdateable naiveBayesUpdateable, J48 j48, DecisionTable decisionTable) {
        long tempoinicial = System.currentTimeMillis();

        /**
         * criacao dos atributos de definicoes para a obtencao da matrix de
         * confusao
         */
        String predicao, atual = null;
        double pred = 0;
        int vp = 0, vn = 0, fp = 0, fn = 0, total_instancias = 0;

        try {

            if (naiveBayesUpdateable != null) {
                naiveBayesUpdateable.buildClassifier(instanciatreino);
            }
            if (j48 != null) {
                j48.setUnpruned(true);
                j48.buildClassifier(instanciatreino);
            }
            if (decisionTable != null) {
                decisionTable.buildClassifier(instanciatreino);
            }

            for (int i = 1; i <= instanciateste.numInstances() - 1; i++) {
                if (naiveBayesUpdateable != null) {
                    pred = naiveBayesUpdateable.classifyInstance(instanciateste.instance(i));
                }
                if (j48 != null) {
                    pred = j48.classifyInstance(instanciateste.instance(i));
                }
                if (decisionTable != null) {
                    pred = decisionTable.classifyInstance(instanciateste.instance(i));
                }

                atual = instanciateste.classAttribute().value((int) instanciateste.instance(i).classValue());
                predicao = instanciateste.classAttribute().value((int) pred);
                System.out.print("ID: " + i + ", atual: " + atual + ", predicao: " + predicao + "\n");

                /**
                 * obtencao do verdadeiro positivo. Se o atual for "R" e a
                 * predicao for "R", entao e esperado um verdadeiro positivo.
                 *
                 */
                if (!atual.equalsIgnoreCase("normal.") && !predicao.equalsIgnoreCase("normal.")) {
                    vp++;
                }
                /**
                 * obtencao do verdadeiro negativo. Se o atual não for "R" e a
                 * predicao não for "R", entao e esperado um verdadeiro negativo
                 *
                 */
                if (atual.equalsIgnoreCase("normal.") && predicao.equalsIgnoreCase("normal.")) {
                    vn++;
                }

                /**
                 * Sobre a condicao para o falso positivoe e esperado que o *
                 * atual nao seja um "R" e a predicao esperada ocorra um "R". "
                 */
                if (atual.equalsIgnoreCase("normal.") && !predicao.equalsIgnoreCase("normal.")) {
                    fp++;
                }

                /**
                 * Sobre a condicao para o falso negativo e esperado que o *
                 * atual seja um "R" e a predicao esperada ocorra um "R". "
                 */
                if (!atual.equalsIgnoreCase("normal.") && predicao.equalsIgnoreCase("normal.")) {
                    fn++;
                }

                total_instancias++;
            }

            double acuracia = (vp * 100.00) / (vp + vn);
            double precisao = (vp * 100.00) / (vp + fp);
            double recall = (vp * 100.00) / (vp + fn);

            System.out.print(" total de instancias : " + total_instancias);
            System.out.print("\n verdadeiro positivo : " + vp + "\n verdadeiro negativo : " + vn);
            System.out.print("\n falso positivo : " + fp + "\n falso negativo : " + fn);
            System.out.printf("\n precisao : %.2f%n ", precisao);
            System.out.printf("recall :   %.2f%n ", recall);
            System.out.printf("acuracia : %.2f%n ", acuracia);

        } catch (Exception ex) {
            Logger.getLogger("IDS logger").log(Level.SEVERE, null, ex);
            if (ex instanceof IllegalArgumentException) {
                javax.swing.JOptionPane.showMessageDialog(null,
                        "Houve erro com o teste realizado!!!",
                        "Erro",
                        javax.swing.JOptionPane.ERROR_MESSAGE);
            }
        }
        long tempofinal = System.currentTimeMillis();

        System.out.printf("Tempo decorrido : %.3f ms %n", (tempofinal - tempoinicial) / 1000d);
        dataEHora();

        javax.swing.JOptionPane.showMessageDialog(null,
                "Teste realizado com sucesso",
                "Sucesso!!!",
                javax.swing.JOptionPane.INFORMATION_MESSAGE);
    }

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        Logger.getLogger("IDS logger").info("Treinamento de classificadores em Machine Learning ");

        try {

            Instances instanciatreino = DataSource.read("/matheus/nids/arquivotreinoeteste/treinoeteste/varias_anomalias/Train.arff");
            Instances instanciateste = DataSource.read("/matheus/nids/arquivotreinoeteste/treinoeteste/varias_anomalias/Test.arff");

            instanciatreino.setClassIndex(instanciatreino.numAttributes() - 1);
            instanciateste.setClassIndex(instanciateste.numAttributes() - 1);

            if (!instanciatreino.equalHeaders(instanciateste)) {
                Logger.getLogger("IDS logger").info("DataSets sao incompativeis!!!");
                throw new IllegalArgumentException("DataSets sao incompativeis ");
            }

            NaiveBayesUpdateable naiveBayes = new NaiveBayesUpdateable();
            J48 j48 = new J48();
            DecisionTable decisionTable = new DecisionTable();

            Remove remove = new Remove();
            remove.setAttributeIndices("1");

            int op = -1;

            escolhas();
            op = scan.nextInt();

            mensagem(op);

            while (op != 0) {

                if (op == 1) {
                    Logger.getLogger("IDS logger").info("Treinamento de classificador NaiveBayes");
                    classificacao(instanciatreino, instanciateste, naiveBayes, null, null);
                }

                if (op == 2) {
                    Logger.getLogger("IDS logger").info("Treinamento de classificador Arvore de Decisao");
                    classificacao(instanciatreino, instanciateste, null, j48, null);
                }
                if (op == 3) {
                    Logger.getLogger("IDS logger").info("Treinamento de classificador Tabela de Decisao");
                    classificacao(instanciatreino, instanciateste, null, null, decisionTable);
                }
                escolhas();
                op = scan.nextInt();
                mensagem(op);
            }

        } catch (Exception ex) {
            Logger.getLogger("ARS logger").log(Level.SEVERE, null, ex);
            if (ex instanceof IllegalArgumentException) {
                javax.swing.JOptionPane.showMessageDialog(null,
                        "O arquivo de treinamento/teste fornecido (.ARFF) e invalido", "Erro",
                        javax.swing.JOptionPane.ERROR_MESSAGE);
            }
        }
    }

}

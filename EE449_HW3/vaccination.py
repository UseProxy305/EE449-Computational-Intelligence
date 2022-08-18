import os
from matplotlib import pyplot as plt
from matplotlib import ticker

# class object to simulate vaccinating people
class Vaccination(object):

    def __init__(self):

        self.vaccinated_percentage_curve_ = [0.] # percentage of the vaccinated people
        self.vaccination_rate_curve_ = [0.]   # percentage / day
        self.vaccination_control_curve_ = [0.]    # percentage / day

    def _vaccinationFailureRate(self):

        p = self.vaccinated_percentage_curve_[-1]

        return .5 * p * p

    # method to apply the control
    def vaccinatePeople(self, vaccination_control):
        """
    applies the control signal to vaccinate people and updates the status curves

    Arguments:
    ----------

    vaccination_control: float, vaccination rate to be added to the current vaccination rate

        """

        # update vaccination rate according to control signal
        curr_vaccination_rate = self.vaccination_rate_curve_[-1]
        vaccination_rate = max(0., min(.6, curr_vaccination_rate + vaccination_control))

        effective_vaccination_rate = vaccination_rate - self._vaccinationFailureRate()

        # update the vaccinated percentage after 0.1 Day of vaccination with the current rate
        vaccination_percentage = \
            min(1., self.vaccinated_percentage_curve_[-1] + effective_vaccination_rate * .1)

        # update status curves
        self.vaccinated_percentage_curve_.append(vaccination_percentage)
        self.vaccination_rate_curve_.append(vaccination_rate)
        self.vaccination_control_curve_.append(vaccination_control)

    # method to obtain measurements
    def checkVaccinationStatus(self):
        """
    returns the current vaccinated percentage and vaccination rate as a two-tuple
    (vaccinated_percentage, vaccination_rate)

    Returns:
    ----------

    (vaccinated_percentage, effective_vaccination_rate): (float, float) tuple,
                    vaccination percentage and rate to be used by the controller

        """

        vaccinated_percentage = self.vaccinated_percentage_curve_[-1]
        effective_vaccination_rate = \
            self.vaccination_rate_curve_[-1] - self._vaccinationFailureRate()

        return (vaccinated_percentage, effective_vaccination_rate)

    # method to visualize the results for the homework
    def viewVaccination(self, point_ss, vaccination_cost, save_dir='', filename='vaccination', show_plot=True):
        """
        plots multiple curves for the vaccination and
            saves the resultant plot as a png image

        Arguments:
        ----------

        point_ss: int, the estimated iteration index at which the system is at steady state

        vaccination_cost: float, the estimated cost of the vaccination until the steady state

        save_dir: string, indicating the path to directory where the plot image is to be saved

        filename: string, indicating the name of the image file. Note that .png will be automatically
        appended to the filename.

        show_plot: bool, whether the figure is to be shown

        Example:
        --------

        visualizing the results of the vaccination

        # assume many control signals have been consecutively applied to vaccine people

        >>> my_vaccine = Vaccination()

        >>> my_vaccine.vaccinatePeople(vaccination_control) # assume this has been repeated many times

        >>> # assume state state index and the vaccination cost have been computed

        >>> # as point_ss=k and vaccination_cost=c

        >>> my_vaccine.viewVaccination(point_ss=k, vaccination_cost=c,
        >>>                      save_dir='some\location\to\save', filename='vaccination')

        """

        color_list = ['#ff0000', '#32CD32', '#0000ff', '#d2691e', '#ff00ff', '#000000', '#373788']
        style_list = ['-', '--']

        num_plots = 3

        plot_curve_args = [{'c': color_list[k],
                            'linestyle': style_list[0],
                            'linewidth': 3} for k in range(num_plots)]

        plot_vert_args = [{'c': color_list[k],
                            'linestyle': style_list[1],
                            'linewidth': 3} for k in range(num_plots)]

        font_size = 18

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        day_x = [i * .1 for i in range(len(self.vaccinated_percentage_curve_))]
        x_ticks = day_x[::10]

        # vaccinated population
        ax = axes[0]
        ax.set_title('vaccinated population percentage over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss+1], self.vaccinated_percentage_curve_[:point_ss+1], **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccinated_percentage_curve_[point_ss:], **plot_curve_args[1])
        ax.plot([day_x[point_ss]] * 2, [0, self.vaccinated_percentage_curve_[point_ss]], **plot_vert_args[2])


        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccinated population %', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, lw = 1, ls = '--', c = '.75')

        # vaccination rate
        ax = axes[1]
        ax.set_title('vaccination rate over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss + 1], self.vaccination_rate_curve_[:point_ss + 1],
                **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccination_rate_curve_[point_ss:],
                **plot_curve_args[1])
        ax.plot([day_x[point_ss]] * 2, [0, self.vaccination_rate_curve_[point_ss]],
                **plot_vert_args[2])
        ax.fill_between(day_x[:point_ss + 1], 0, self.vaccination_rate_curve_[:point_ss + 1],
                        facecolor='#FF69B4', alpha=0.7)

        ax.text(1.5, .01, 'cost = %.2f'%vaccination_cost,
                horizontalalignment='center', fontsize=font_size)

        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccination rate (%/day)', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, lw=1, ls='--', c='.75')

        # vaccination rate control
        ax = axes[2]
        ax.set_title('vaccination rate control over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss + 1], self.vaccination_control_curve_[:point_ss + 1],
                **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccination_control_curve_[point_ss:],
                **plot_curve_args[1])
        y_min = ax.get_ylim()[0]
        ax.plot([day_x[point_ss]] * 2,
                [y_min, self.vaccination_control_curve_[point_ss]],
                **plot_vert_args[2])

        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccination rate control (%/day)', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=y_min)
        ax.grid(True, lw=1, ls='--', c='.75')

        if show_plot:
            plt.show()

        fig.savefig(os.path.join(save_dir, filename + '.png'))